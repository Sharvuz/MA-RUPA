

### 1. losses.py: Anti-Saturation & Dynamic Margin

**Mathematics:**
Standard MIL suffers from gradient saturation when $S_{anom} \gg S_{norm}$. We introduce entropy-weighted scaling $\mathcal{H}$ and a decaying margin $m_t$:
$$ \mathcal{L}_{AS-MIL} = \mathcal{H}(p) \cdot \max(0, m_t - (S_{anom} - S_{norm})) $$
$$ m_t = m_0 \cdot e^{-\gamma \cdot epoch}, \quad \mathcal{H}(p) = -\sum p \log p $$

**Code:**
```python
class AntiSaturationMILLoss(nn.Module):
    def __init__(self, m0=1.0, gamma=0.01):
        super().__init__()
        self.m0, self.gamma = m0, gamma

    def forward(self, s_anom, s_norm, epoch):
        margin = self.m0 * np.exp(-self.gamma * epoch)
        diff = s_anom - s_norm
        # Entropy weighting to prevent sink
        p = torch.sigmoid(diff)
        entropy = -(p * torch.log(p + 1e-8) + (1-p) * torch.log(1-p + 1e-8))
        return torch.mean(entropy * torch.relu(margin - diff))
```

### 2. model.py: ASPM & Dynamic Routing

**Mathematics:**
ASPM decomposes features $f$ into reconstructible normal patterns $f_{rec}$ and anomalous residuals $r_{anom}$:
$$ f_{rec} = \mathcal{D}(\mathcal{E}(f)), \quad r_{anom} = f - f_{rec} $$
Dynamic Routing Gate $g$ adapts based on residual magnitude:
$$ g = \sigma(W_g [r_{anom} \oplus f_{rec}] + b_g), \quad f_{out} = g \cdot r_{anom} + (1-g) \cdot f_{rec} $$

**Code:**
```python
class ASPMModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Linear(dim, dim//2)
        self.decoder = nn.Linear(dim//2, dim)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())

    def forward(self, f):
        f_rec = self.decoder(self.encoder(f))
        r_anom = f - f_rec
        g = self.gate(torch.cat([r_anom, f_rec], dim=-1))
        return g * r_anom + (1 - g) * f_rec, r_anom
```

### 3. ucf_train.py: Loss Balancing

**Strategy:**
Dynamic weighting prioritizes consistency early, shifting to CLASM (Contrastive Local Anomaly Score Matching) later.
$$ \mathcal{L}_{total} = \lambda_1(t)\mathcal{L}_{CLASM} + \lambda_2(t)\mathcal{L}_{cons} + \lambda_3\mathcal{L}_{anchor} $$

**Code:**
```python
def get_weights(epoch, total_epochs):
    # Shift focus from consistency to anomaly discrimination
    lam_clasm = 0.5 + 0.5 * (epoch / total_epochs)
    lam_cons = 1.0 - (epoch / total_epochs)
    lam_anchor = 0.1
    return lam_clasm, lam_cons, lam_anchor

# Inside training loop
l_c, l_con, l_a = get_weights(epoch, epochs)
loss = l_c * loss_clasm + l_con * loss_cons + l_a * loss_anchor
```

### 4. ucf_test.py: Inference Fusion

**Strategy:**
To achieve $\ge 89.5\%$ AUC, fuse anomaly score $S_{anom}$ with reconstruction error $E_{rec}$ using learned coefficients $\alpha, \beta$:
$$ S_{final} = \alpha \cdot S_{anom} + \beta \cdot \|r_{anom}\|_2 $$

**Code:**
```python
def compute_final_score(s_anom, r_anom, alpha=0.7, beta=0.3):
    # Normalize residual norm to match score scale
    rec_err = torch.norm(r_anom, p=2, dim=-1)
    rec_err = (rec_err - rec_err.mean()) / (rec_err.std() + 1e-8)
    return alpha * s_anom + beta * rec_err
```