import os

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

P_STRONG = _get_float("P_STRONG", 0.70)
P_WEAK = _get_float("P_WEAK", 0.45)
S_VERY_HIGH = _get_float("S_VERY_HIGH", 0.86)
S_HIGH = _get_float("S_HIGH", 0.78)
K_MIN = int(os.getenv("K_MIN", "2"))

USE_DUMMY_CLASSIFIER = os.getenv("USE_DUMMY_CLASSIFIER", "true").lower() == "true"


DICT_TERMS = {
    "Device Malfunction": [
        # General device failure
        "device malfunction", "equipment malfunction", "malfunction during use",
        "device not working", "device failed", "failure during operation",
        "failure to function", "malfunction on deployment", "malfunction after use",

        # Deployment / operational issues
        "deployment failure", "incomplete deployment", "unable to deploy",
        "failed to deploy", "partial deployment", "deployment interrupted",
        "failed to lock", "locking issue", "activation failure",
        "device misfire", "delivery issue", "misalignment of device",

        # Structural / mechanical failures
        "device damage", "component breakage", "handle breakage",
        "material integrity issue", "broken tip", "torn sleeve",
        "kinked sheath", "catheter kink", "component dislodged",
        "sealant exposed", "loose seal", "sealant leakage", "sealant failure",

        # Blockages and leaks
        "balloon leak", "balloon burst", "balloon deflation",
        "unable to inflate", "leak in system", "fluid leakage",

        # Obstruction / entrapment
        "device stuck", "entrapment issue", "obstructed device",
        "retained device", "unable to remove device",

        # Electronics / software
        "system error", "software error", "firmware bug",
        "power failure", "battery failure", "low battery alert",
        "display malfunction", "false alarm", "incorrect reading",


        "cuff miss", "quality issue", "reported difficulty", "device returned", 
        "device behavior", "device performance", "reported malfunction", 
        "returned for analysis", "proglide failure", "mechanical issue",
        "clicking sound", "double click", "clicks on activation",
        "device click", "unexpected click", "plunger click",
        "audible click", "strange sound", "device noise", 
        "mechanical feedback", "anchor failed", "footplate broken", 
        "collagen plug issue", "plug misplacement", "clip misfire", 
        "clip dislodged", "anchor not seated", "prostar failure",
        "device mispositioned", "poor seal", "incomplete closure", 
        "deployment drift", "suture break", "clip snapped", "cord rupture", 
        "vcd jammed", "advancer stuck", "release failure", 
        "footing not deployed", "clip ejected", "seal not achieved", "device knot",
        "deployment issue", "catheter jammed", "stuck device"
    ],

    "Injury": [
        # Bleeding / hematoma
        "bleeding", "excessive bleeding", "control of bleeding",
        "inability to stop bleeding", "hemostasis failure", "manual compression",
        "hematoma", "hematoma formation", "site hematoma",

        # Pain & discomfort
        "pain", "site pain", "localized pain", "post-procedure pain",
        "site discomfort", "discomfort", "local irritation",

        # Infections & inflammation
        "infection", "site infection", "wound infection",
        "inflammatory reaction", "swelling", "redness",
        "bruising", "seroma", "delayed healing", "wound complication",

        # Vascular / tissue injuries
        "vessel injury", "vascular injury", "arterial injury",
        "tissue damage", "tissue necrosis", "necrosis",
        "pseudoaneurysm", "extravasation", "vascular spasm",
        "ischemia", "limb ischemia", "nerve injury", "limb numbness", "numbness",

        # Hypersensitivity / allergy
        "hypersensitivity reaction", "allergic reaction", "skin reaction",
        "contact dermatitis", "rash", "burns",

        # Extra from your extended set
        "hematosis", "occlusion", "retroperitoneal bleed", "av fistula", 
        "femoral pseudoaneurysm", "vessel perforation", "artery dissection", 
        "groin pain", "groin hematoma", "thrombus formation", 
        "deep vein thrombosis", "vascular laceration", "prolonged bleeding", 
        "puncture site complication", "arterial embolism","skin reaction"
    ],

    "Death": [
        # Direct death reports
        "patient died", "patient death", "death reported",
        "death occurred", "fatal event", "fatal outcome",
        "loss of life", "mortality", "fatality", "lethal event",
        "procedure-related death", "unexpected death", "sudden death",
        "death following procedure", "fatal complication",

        # Common MAUDE phrasings
        "patient demise", "passed away", "found deceased",
        "collapsed and died", "expired", "death confirmed",
        "pronounced dead", "declared dead", "cause of death",
        "autopsy revealed", "post mortem", "post-mortem",
        "death certificate", "no signs of life", "end of life",

        # Indirect death phrasing
        "died suddenly", "died during procedure", "died after procedure",


        "cardiac arrest during procedure", "exsanguination", "massive bleed",
        "vascular collapse", "procedure fatality", "died from complication",
        "bleed-out", "femoral rupture", "death due to AV fistula", 
        "fatal stroke", "multi-organ failure", "cardiovascular collapse"
    ]
}