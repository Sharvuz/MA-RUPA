"""Canonical Concept + LAS-VAD Anomaly Attribute Clues (AAT) for CLIP Vision-Language WS-VAD.

Synthesizes:
  1. VadCLIP / DSANet Canonical Concept Anchor (Index 0: `X_lang`), free of shared template boilerplate
     ("a surveillance video of...") so `text_orthogonality_loss` (`loss3`) converges cleanly to 0.0000.
  2. LAS-VAD (Wang & Zhao, arXiv:2603.00550, Sec. 3.1 & Fig. 1) Observable Sensory Attribute Clues
     (Index 1..K: `X_aux`), capturing distinct visual and acoustic physical signatures (e.g., flames,
     thick smoke, bright flash, flying debris for explosion/arson; rapid aggressive punching/grappling
     for fighting/assault) to eliminate 100% miss rates on subtle/sudden anomalies.
"""

DESCRIPTIONS_ORI = {
    "normal": [
        "normal",
        "regular surroundings, no chaotic elements, peaceful calm scene",
        "orderly pedestrian walking, steady routine traffic, calm daily activity",
    ],
    "abuse": [
        "abuse",
        "repeated striking hitting motion, victim recoiling shielding face, cruel physical mistreatment",
        "aggressive dominant posture, striking helpless victim, violent physical abuse",
    ],
    "arrest": [
        "arrest",
        "police officer restraining suspect, handcuffing motion, suspect pinned against car or ground",
        "law enforcement officers surrounding detainee, flashing police lights, physical apprehension",
    ],
    "arson": [
        "arson",
        "deliberate fire ignition, pouring accelerant liquid, sudden flame spreading from point source",
        "expanding fire blaze, thick rising smoke plume, burning property or vehicle at night",
    ],
    "assault": [
        "assault",
        "sudden forceful physical attack, punching kicking victim knocked to ground",
        "violent street attack, swinging weapon at victim, injured person collapsing on pavement",
    ],
    "burglary": [
        "burglary",
        "forced entry through door or window, prying lock, breaking glass at building entrance",
        "intruder sneaking into dark room or store, ransacking drawers, carrying stolen goods",
    ],
    "explosion": [
        "explosion",
        "flames, thick smoke, flying debris, damaged structures, bright flash at blast site",
        "sudden explosive fireball ignition, shockwave dust cloud, shattered debris scattering outward",
    ],
    "fighting": [
        "fighting",
        "rapid aggressive physical brawl between multiple people, punching kicking grappling",
        "chaotic group fistfight, bodies colliding and falling, violent physical struggle",
    ],
    "roadaccidents": [
        "roadaccidents",
        "high speed vehicle collision, car crash impact deformation, skidding tires and flying debris",
        "pedestrian or motorcycle struck by car, rollover crash on road, sudden traffic collision",
    ],
    "robbery": [
        "robbery",
        "armed holdup threatening victim with weapon, demanding money or valuables by force",
        "perpetrator snatching bag or cash register, violent mugging and rapid escape",
    ],
    "shooting": [
        "shooting",
        "aimed firearm muzzle flash and recoil, gunshot impact, victim collapsing to ground",
        "gunman firing weapon, people fleeing in panic, armed shooting attack",
    ],
    "shoplifting": [
        "shoplifting",
        "furtively concealing store merchandise inside clothing or bag, looking around nervously",
        "secretly pocketing retail item from shelf, bypassing checkout counter without paying",
    ],
    "stealing": [
        "stealing",
        "quickly grabbing unattended property bicycle or bag, furtive theft and running away",
        "pickpocketing or stealing vehicle from parked area, rapid illicit taking of object",
    ],
    "vandalism": [
        "vandalism",
        "smashing window glass, breaking public property, kicking or damaging parked car",
        "deliberate destruction of property, spray painting graffiti, overturning street objects",
    ],
}

DESCRIPTIONS_ORI_XD = {
    "A": [
        "normal",
        "regular surroundings, peaceful calm scene, normal conversation and steady movement",
        "orderly pedestrians and routine traffic, calm background without violence",
    ],
    "B1": [
        "fighting",
        "rapid aggressive physical combat, punching kicking grappling, violent brawl",
        "hand to hand fight, bodies slamming and falling, impact thuds and shouting",
    ],
    "B2": [
        "shooting",
        "aimed gun muzzle flash, firearm recoil, gunshot crack and victim falling",
        "armed shootout, firing pistol or rifle, bullet impacts and panic screaming",
    ],
    "B4": [
        "riot",
        "chaotic crowd violence, mob throwing projectiles, street clashes with smoke and fire",
        "violent protest disorder, crowd charging and smashing barriers, loud crowd uproar",
    ],
    "B5": [
        "abuse",
        "cruel physical beating, striking helpless victim, victim recoiling and crying out",
        "violent slapping kicking or whipping, aggressive physical torment",
    ],
    "B6": [
        "car accident",
        "violent vehicle crash collision, screeching tires, metal impact and shattered glass",
        "car smashing into obstacle or vehicle rollover, road crash debris scattering",
    ],
    "G": [
        "explosion",
        "flames, thick smoke, flying debris, damaged structures, bright flash at blast site",
        "explosive fireball blast wave, loud boom and collapsing structure in smoke",
    ],
}
