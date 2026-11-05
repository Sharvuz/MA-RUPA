"""Canonical Concept Prompts for Video-Conditioned CLIP Vision-Language WS-VAD.

Follows VadCLIP (Wu et al., AAAI 2024, Table 4), DSANet, and the Argon-4 Gradient Conservation Law:
Bare canonical class concepts outperform long sentence templates because shared template words
inflate Normal-vs-Abnormal cosine similarity inside the CLIP cone and dilute Top-k MIL gradients
on subtle/fast-onset classes ("arson", "explosion", "roadaccidents", "shooting").
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
