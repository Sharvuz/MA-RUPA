import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.losses import (
    CLAS2,
    CLASM,
    CLASM_EVENT,
    RUPA_RESIDUAL_EVENT,
    CLASM_BKG,
    text_orthogonality_loss,
    ConsistencyLoss,
    routing_anchor_loss,
    edge_preserving_smoothness_loss,
    audio_guided_mil_loss,
    iam_intention_loss,
    acc_pseudo_label_loss,
    WarmCosineScheduler,
    ModelEMA,
)


def test_interleaved_dataparallel_distribution():
    print("Testing Interleaved DataParallel Batch Distribution...")
    B_half = 4
    T, D = 16, 512

    norm_feat = torch.full((B_half, T, D), 10.0)
    anom_feat = torch.full((B_half, T, D), 20.0)

    visual_batch = torch.empty((2 * B_half, T, D), dtype=norm_feat.dtype)
    visual_batch[0::2] = norm_feat
    visual_batch[1::2] = anom_feat

    binary_labels = torch.empty(2 * B_half, dtype=torch.float32)
    binary_labels[0::2] = 0.0
    binary_labels[1::2] = 1.0

    gpu0_labels = binary_labels[:B_half]
    gpu1_labels = binary_labels[B_half:]

    assert (gpu0_labels == 0.0).sum().item() == B_half // 2
    assert (gpu0_labels == 1.0).sum().item() == B_half // 2
    assert (gpu1_labels == 0.0).sum().item() == B_half // 2
    assert (gpu1_labels == 1.0).sum().item() == B_half // 2
    print("  Interleaved DataParallel distribution passed!")


def test_sota_losses_and_gradients():
    print("Testing MA-RUPA v3 Loss Suite & Gradient Flow...")
    B, T, D, C = 4, 32, 512, 14
    device = torch.device("cpu")

    logits1 = torch.randn(B, T, 1, requires_grad=True)
    logits2 = torch.randn(B, T, C, requires_grad=True)
    logits3 = torch.randn(B, 1, C, requires_grad=True)
    logits4 = torch.randn(B, 1, C, requires_grad=True)
    aux_logits = torch.randn(B, T, 1, requires_grad=True)
    text_features = torch.randn(C, D, requires_grad=True)

    lengths = torch.tensor([32, 28, 20, 30])
    labels = torch.zeros(B, C)
    labels[0, 0] = 1.0  # Normal
    labels[1, 3] = 1.0  # Arson
    labels[2, 0] = 1.0  # Normal
    labels[3, 7] = 1.0  # Fighting

    aux_sem_logits = torch.randn(B, T, C, requires_grad=True)
    iam_logits = torch.randn(B, T, C, requires_grad=True)
    iam_prototypes = torch.randn(C, D)
    acc_pseudo = torch.softmax(torch.randn(B, T, C), dim=-1)
    fine_grained_probs = torch.softmax(logits2, dim=-1)

    l1 = CLAS2(logits1, labels, lengths, device)
    l2 = CLASM(logits2, labels, lengths, device)
    l3 = text_orthogonality_loss(text_features)
    l4 = RUPA_RESIDUAL_EVENT(logits3, labels, lengths, device)
    l4_ev = CLASM_EVENT(logits3, labels, lengths, device)
    l5 = CLASM_BKG(logits4, labels, lengths, device)
    l_aux = audio_guided_mil_loss(aux_logits, labels, lengths, aux_sem_logits=aux_sem_logits)
    l_sm = edge_preserving_smoothness_loss(torch.sigmoid(logits1), lengths)

    dnp_mock = {
        'iam_logits': iam_logits,
        'iam_scores': 1.0 - torch.softmax(iam_logits, dim=-1)[:, :, :1],
        'acc_pseudo_labels': acc_pseudo,
        'fine_grained_probs': fine_grained_probs,
        'detector_scores': torch.sigmoid(logits1),
    }
    l_iam = iam_intention_loss(dnp_mock, labels, lengths, device, iam_prototypes=iam_prototypes)
    l_acc = acc_pseudo_label_loss(dnp_mock, lengths, logits1=logits1)

    f_orig = torch.randn(B, T, D, requires_grad=True)
    f_rec = torch.randn(B, T, D, requires_grad=True)
    cons_fn = ConsistencyLoss()
    l_cons = cons_fn(logits1, f_orig, f_rec, lengths)

    routing_scores = torch.sigmoid(logits1 * 0.9)
    l_anchor = routing_anchor_loss(
        {'routing_scores': routing_scores, 'detector_scores': torch.sigmoid(logits1)},
        lengths,
    )

    total = (
        l1 + 1.1 * l2 + l3 + 0.2 * l4 + 0.05 * l4_ev + 0.2 * l5
        + 0.25 * l_aux + 0.4 * l_iam + 0.3 * l_acc + 0.02 * l_sm + 0.1 * l_cons + 0.1 * l_anchor
    )
    assert total.dim() == 0 and torch.isfinite(total)
    total.backward()

    for name, tensor in [
        ('logits1', logits1), ('logits2', logits2), ('logits3', logits3),
        ('logits4', logits4), ('aux_logits', aux_logits), ('aux_sem_logits', aux_sem_logits),
        ('iam_logits', iam_logits), ('text_features', text_features),
    ]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), f"Bad grad on {name}"

    print("  All SOTA loss functions & gradients passed!")


def test_ema_and_warm_cosine_scheduler():
    print("Testing ModelEMA & WarmCosineScheduler...")
    net = torch.nn.Linear(16, 4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    sched = WarmCosineScheduler(opt, base_value=1e-4, final_value=1e-5, total_iters=20, warmup_iters=5)
    ema = ModelEMA(net, decay=0.9)

    for _ in range(10):
        loss = net(torch.randn(2, 16)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update(net)

    orig_w = net.weight.detach().clone()
    backup = ema.apply_to(net)
    assert not torch.allclose(net.weight, orig_w)
    ema.restore(net, backup)
    assert torch.allclose(net.weight, orig_w)
    print("  ModelEMA swap/restore & WarmCosineScheduler passed!")


if __name__ == '__main__':
    test_interleaved_dataparallel_distribution()
    test_sota_losses_and_gradients()
    test_ema_and_warm_cosine_scheduler()
    print("\n[SUCCESS] ALL TRAINING PIPELINE LOSS TESTS PASSED!")
