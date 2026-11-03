"""Clean, Domain-Grounded Concept Prompts for CLIP Vision-Language WS-VAD.

Follows VadCLIP (AAAI 2024), TPWNG (CVPR 2024), and DSANet principles:
Concise canonical category names paired with crisp domain anchors so CLIP's text
adapter learns orthogonal normal-vs-anomaly boundaries without prompt dilution.
"""

DESCRIPTIONS_ORI = {
    "normal": [
        "normal",
        "a normal peaceful surveillance video with routine activity",
    ],
    "abuse": [
        "abuse",
        "a surveillance video of violent physical abuse and mistreatment",
    ],
    "arrest": [
        "arrest",
        "a surveillance video of police officers arresting a suspect",
    ],
    "arson": [
        "arson",
        "a surveillance video of arson fire flames and thick smoke",
    ],
    "assault": [
        "assault",
        "a surveillance video of a sudden violent street assault attack",
    ],
    "burglary": [
        "burglary",
        "a surveillance video of a burglar breaking into a building or shop",
    ],
    "explosion": [
        "explosion",
        "a surveillance video of a sudden bright explosion blast and smoke",
    ],
    "fighting": [
        "fighting",
        "a surveillance video of people fighting and punching in a brawl",
    ],
    "roadaccidents": [
        "road accident",
        "a surveillance video of a vehicle collision car crash road accident",
    ],
    "robbery": [
        "robbery",
        "a surveillance video of an armed robbery and violent holdup",
    ],
    "shooting": [
        "shooting",
        "a surveillance video of a gunman shooting a firearm weapon",
    ],
    "shoplifting": [
        "shoplifting",
        "a surveillance video of shoplifting and hiding merchandise in a store",
    ],
    "stealing": [
        "stealing",
        "a surveillance video of a thief stealing property or a vehicle",
    ],
    "vandalism": [
        "vandalism",
        "a surveillance video of vandalism smashing and destroying property",
    ],
}

DESCRIPTIONS_ORI_XD = {
    "A": [
        "normal",
        "a peaceful normal video scene without violence or danger",
    ],
    "B1": [
        "fighting",
        "a violent video scene of people fighting punching and brawling",
    ],
    "B2": [
        "shooting",
        "a violent video scene of gunfire shooting weapons and shootouts",
    ],
    "B4": [
        "riot",
        "a chaotic video scene of a violent street riot and crowd clash",
    ],
    "B5": [
        "abuse",
        "a violent video scene of physical abuse beating and cruelty",
    ],
    "B6": [
        "car accident",
        "a video scene of a high impact car crash and vehicle collision",
    ],
    "G": [
        "explosion",
        "a video scene of a fiery bomb explosion blast and smoke",
    ],
}
