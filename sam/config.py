"""
Shared config for the PWTT-CNN pipeline: per-city war_start dates and the
spatial train / val / test split. Both export_s1_patches.py and
train_siamese.py import from here so the split is defined in exactly one place.
"""
import os

# ---------------------------------------------------------------------------
# war_start per city.
#   Ukraine (all cities) and Gaza are authoritative — taken from PWTT's eval.py.
#   The 2016-17 Middle East dates are best estimates: a clean 12-month pre-war
#   Sentinel-1 baseline is hard there because S1 only starts Oct 2014 AND these
#   cities saw years of prior conflict, so "12 months before annotation" may
#   already contain damage. That's why they're an OOD test set below, not train.
# ---------------------------------------------------------------------------
_UKRAINE = "2022-02-22"
_WAR_START = {
    "Gaza":   "2023-10-10",
    "Mosul":  "2016-10-16",   # Battle of Mosul began
    "Raqqa":  "2017-06-06",   # SDF offensive began
    "Aleppo": "2016-07-01",   # final regime offensive; baseline most compromised
}

def war_start_for(city):
    return _WAR_START.get(city, _UKRAINE)

def parse_name(csv_path):
    """'Makariv_20220316_1_footprints.csv' -> ('Makariv', '2022-03-16').
    The filename date is the UNOSAT annotation date = our inference_start."""
    base = os.path.basename(csv_path)
    parts = base.split("_")
    city, date = parts[0], parts[1]
    inference_start = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    return city, inference_start

# ---------------------------------------------------------------------------
# Spatial split — WHOLE cities, so no building leaks between splits.
# Chosen to spread damage rate (0.08% .. 57%) and country across every split:
#  - big, high-signal cities (Gaza, Rubizhne, Sievierodonetsk) stay in TRAIN so
#    the encoder actually sees the damage signature;
#  - VAL and TEST each span the full imbalance range, including the brutal
#    sub-1% cities where precision is hardest;
#  - the 3 Middle East cities are a separate OOD test (baseline caveat above).
# Edit these freely — it's the only place the split is defined.
# ---------------------------------------------------------------------------
SPLIT = {
    # TRAIN
    "Gaza": "train", "Rubizhne": "train", "Sievierodonetsk": "train",
    "Hostomel": "train", "Avdiivka": "train", "Lysychansk": "train",
    "Makariv": "train", "Bucha": "train", "Trostianets": "train",
    "Mykolaiv": "train", "Kremenchuk": "train", "Antonivka": "train",
    "Kherson": "train",
    # VAL  (early stopping, threshold selection)
    "Irpin": "val", "Chernihiv": "val", "Shchastia": "val", "Kramatorsk": "val",
    # TEST (in-distribution, report once at the very end)
    "Mariupol": "test", "Kharkiv": "test", "Okhtyrka": "test",
    "Melitopol": "test", "Sumy": "test",
    # OOD TEST (cross-theatre generalization; report separately)
    "Aleppo": "ood", "Raqqa": "ood", "Mosul": "ood",
}

def split_of(city):
    return SPLIT.get(city)