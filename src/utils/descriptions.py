"""Canonical Concept Prompts for Video-Conditioned CLIP Vision-Language WS-VAD.

Follows VadCLIP (Wu et al., AAAI 2024, Table 4) & DSANet:
Bare canonical class concepts outperform manual sentence templates ("a surveillance video of...")
when paired with the learned video-conditioned visual prompt (`text_features + visual_attn + mlp1`),
because shared template prefixes artificially inflate Normal-vs-Abnormal cosine similarity and
force `text_orthogonality_loss` (`loss3`) to warp discriminative class semantics.
"""

DESCRIPTIONS_ORI = {
    "normal": ["normal"],
    "abuse": ["abuse"],
    "arrest": ["arrest"],
    "arson": ["arson"],
    "assault": ["assault"],
    "burglary": ["burglary"],
    "explosion": ["explosion"],
    "fighting": ["fighting"],
    "roadaccidents": ["roadaccidents"],
    "robbery": ["robbery"],
    "shooting": ["shooting"],
    "shoplifting": ["shoplifting"],
    "stealing": ["stealing"],
    "vandalism": ["vandalism"],
}

DESCRIPTIONS_ORI_XD = {
    "A": ["normal"],
    "B1": ["fighting"],
    "B2": ["shooting"],
    "B4": ["riot"],
    "B5": ["abuse"],
    "B6": ["car accident"],
    "G": ["explosion"],
}
