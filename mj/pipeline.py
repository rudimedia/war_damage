"""
Shared code for the war damage detection project.

All three notebooks import this file, so the city registry, file paths, data
loading, the models and the augmentation helpers exist in exactly one place.
Upload pipeline.py to the project folder on Google Drive.

Expected folder layout on Google Drive (everything lives under BASE):

    Data/own_footprints/   label CSVs written by notebook 0
    Data/one_month/        published PWTT benchmark
    pipeline.py            this file
    rasters/               Sentinel exports        (written once by notebook 1)
    processed/             patch arrays per city   (written once by notebook 1)
    experiments/           one folder per run      (written by notebook 2)

The division of labour between the notebooks:

    1_preprocessing_eda    downloads imagery ONCE, cuts patches, saves them,
                           and explores the labels and the imagery
    2_training_evaluation  trains a model on the saved patches and evaluates
                           it against the PWTT baseline
    3_interactive_map      applies a trained model and shows the predictions
                           on an interactive map


HOW THE DATA IS STORED (this changed - see MIGRATION at the bottom)
-------------------------------------------------------------------
Three facts drive the layout:

1. The pre-war composite is the SAME for every assessment date, because only
   the post window moves. Storing it once per city instead of once per date
   halves the disk and, more importantly, makes an extra time point cost two
   bands rather than four.
2. Temporal smoothing needs CNN scores at dates that have no labels at all.
   Those dates need post imagery and nothing else.
3. Every array has to stay row-aligned with every other array, or joining a
   pre patch to a post patch silently pairs the wrong buildings.

So each city gets ONE canonical building table, and every patch array is
indexed by position in that table:

    {city}_{tag}_buildings.parquet          canonical rows, all labels
    {city}_{tag}_pre{N}m_X.npy              (n, c_pre, p, p) float16
    {city}_{tag}_pre{N}m_valid.npy          (n,) bool
    {city}_{date}_{tag}_post_{win}_X.npy    (n, c_post, p, p) float16
    {city}_{date}_{tag}_post_{win}_valid.npy

A row exists in every array whether or not its patch is usable; the `valid`
mask says which rows actually hold pixels. Alignment is therefore positional
and free, and a building missing at one date is a masked row rather than a
shifted index.
"""

import os
import re
import glob
import json
import inspect
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import torch
import torch.nn as nn
from shapely.geometry import shape


# --------------------------------------------------------------------------
# Paths. If your Drive folder has a different name, change BASE here and
# in the first cell of each notebook.
# --------------------------------------------------------------------------

BASE = "/content/drive/MyDrive/War-Damage-Detection"
DATA_DIR = os.path.join(BASE, "Data", "own_footprints")   # label CSVs from notebook 0
PWTT_DIR = os.path.join(BASE, "Data", "one_month")        # published benchmark
RASTER_DIR = os.path.join(BASE, "rasters")
# Patches can be large. They stay on Drive by default; set the PATCH_DIR
# environment variable BEFORE importing this module to put them somewhere
# else, e.g. "/content/processed" for Colab's fast local disk.
PROCESSED_DIR = os.environ.get("PATCH_DIR", os.path.join(BASE, "processed"))
EXPERIMENTS_DIR = os.path.join(BASE, "experiments")


# --------------------------------------------------------------------------
# Preprocessing settings. These describe HOW the data on disk was made, so
# they live here and are shared by all notebooks: notebook 1 uses them to
# create the files, notebooks 2 and 3 use them to find the files again.
# Changing anything here means re-running notebook 1.
# --------------------------------------------------------------------------

PREP = {
    # which sensors to export: ["s1"] or ["s1", "s2"]
    "sensors": ["s1"],

    # "footprints" anchors patches on buildings, "grid" on regular cells
    "label_geometry": "footprints",

    # patch side length in pixels (at 10 m per pixel, 32 px = 320 m)
    "patch_size": 32,

    # cap on buildings per city, None keeps every row
    "n_sample": None,  # 4000,
    "seed": 0,

    "imagery": {
        "gee_project": "test-cnn-war-damage",  # your Earth Engine project id
        "pre_months": 12,             # pre war baseline window length
        "post_months": 1,             # post window length, matches "one_month"
        # "backward" looks from the assessment date back, which is what you
        # want when assessments are less than post_months apart
        "post_direction": "backward",
        "scale": 10,                  # metres per pixel
        "aoi_buffer_deg": 0.005,      # padding so edge buildings get a patch
        "s2_max_cloud": 60,           # max cloud percentage for Sentinel 2
        "min_aoi_coverage": 0.90,     # warn if the chosen orbit misses the city
    },

    # ------------------------------------------------------------------
    # Extra, UNLABELLED post windows either side of each assessment date.
    # They exist only so the second-stage model can see a short time series
    # of CNN scores per building. No labels are needed and none of these
    # patches ever enter CNN training.
    #
    # step_days = 12 because that is Sentinel-1's orbital repeat: since S1B
    # failed in 2021 a given relative orbit passes every 12 days, so 12 is the
    # finest grid on which consecutive points contain a different acquisition.
    #
    # window_days is separate on purpose. With the default 1-month window a
    # 12-day shift swaps roughly one scene in three, so neighbouring points
    # are correlated but de-speckled; setting window_days = 12 makes each
    # point a single acquisition, fully distinct but far noisier. Which is
    # better is an empirical question - the XGB feature importances answer it.
    # ------------------------------------------------------------------
    "temporal": {
        "enabled": True,
        "step_days": 12,
        "n_before": 2,
        "n_after": 2,
        "window_days": None,   # None = same length as post_months
    },
}


# --------------------------------------------------------------------------
# City registry. One entry per city.
#
# label_dates: the UNOSAT assessment dates to use, OLDEST FIRST.
# war_start: conflict start date, the pre war imagery window ends here.
# role: "develop" or "holdout". This is about the CITY, not about a split part,
#     and the distinction matters because the word "test" means two different
#     things in this project:
#
#     "develop"  the city the model is built on. Notebook 2 carves it into
#                latitude bands - train / stack / val / test - so it has a
#                test set OF ITS OWN. That in-city test band measures
#                generalisation to unseen GROUND in a city the model knows.
#                Gaza is the only one.
#     "holdout"  never touched by any split part, not even the in-city test
#                band, until the final comparison. These measure something
#                harder and more interesting: generalisation to a whole new
#                CITY, sensor geometry, building stock and conflict.
#
#     So "Gaza is a develop city" and "Gaza:0.00-0.33 is the test band" are
#     both true and not in conflict. Nothing enforces the holdout
#     automatically - notebook 2's split config is written by hand - so
#     held_out_cities() exists to make an accidental inclusion easy to catch.
#
# Two windows have to line up for a city to be usable at all, and both are
# derived from war_start rather than stored:
#
#     pre  = the 12 months ENDING at war_start
#     post = one month BACKWARD from each assessment date
#
# An assessment whose post window starts before war_start is not a
# post-war observation at all: its "pre-war" baseline would be imagery taken
# AFTER the damage, which inverts the comparison the model is built on. And
# Sentinel-1 IW barely exists before 2014-10-01, so nothing earlier can be
# imaged regardless. Notebook 1 checks both and refuses to export a date that
# fails either - see check_city_windows().
# --------------------------------------------------------------------------

CITY_REGISTRY = {
    "Gaza": {
        "label_dates": ["20240503", "20240706", "20240906"],
        "war_start": "2023-10-07",
        "role": "develop",      # split into train/stack/val/test BANDS in nb 2
    },
    # Held-out cities: absent from every split part in notebook 2, including
    # its in-city test band. Labels for all of these come from notebook 0.
    "Raqqa": {
        # UNOSAT publishes five assessments for Raqqa, but only the last one
        # is usable here. 20131022 and 20140212 predate Sentinel-1 entirely,
        # and 20150529 / 20170203 fall INSIDE the pre-war baseline window
        # (2016-06-06 to 2017-06-06), so their "pre-war" imagery would come
        # after the damage. Using an earlier war_start does not rescue them:
        # a 12-month pre window before 2015 has no Sentinel-1 either.
        "label_dates": ["20171021"],
        "war_start": "2017-06-06",
        "role": "holdout",
    },
    "Mosul": {
        # Only the 20170804 release has been processed by notebook 0 so far.
        # CE20140613IRQ_Mosul_damage_assessment.gdb holds three more dates
        # (20170611, 20170616, 20170630) if a temporal series is wanted here.
        "label_dates": ["20170804"],
        "war_start": "2016-10-17",
        "role": "holdout",
    },
    "Chernihiv": {
        "label_dates": ["20220428"],
        "war_start": "2022-02-24",
        "role": "holdout",
    },
    "Rubizhne": {
        "label_dates": ["20220709"],
        "war_start": "2022-02-24",
        "role": "holdout",
    },
}

S1_EARLIEST = "2014-10-01"   # Sentinel-1 IW data barely exists before this


def add_city(name, label_dates, war_start, role="develop"):
    """Register another city, then re-run notebook 1 to preprocess it."""
    CITY_REGISTRY[name] = {"label_dates": list(label_dates),
                           "war_start": war_start, "role": role}


def city_role(city):
    """'develop' or 'holdout'. See the CITY_REGISTRY comment for the split."""
    return CITY_REGISTRY[city].get("role", "develop")


def development_cities():
    """Cities notebook 2 may carve into train / stack / val / test bands.

    Named for the city's role, not a split part: a development city contains
    a test BAND of its own, which is a different thing from a holdout city.
    """
    return [c for c in CITY_REGISTRY if city_role(c) == "develop"]


def held_out_cities():
    """Cities absent from every split part until the final comparison."""
    return [c for c in CITY_REGISTRY if city_role(c) == "holdout"]


# The split parts a model is DEVELOPED on. "test" is in here on purpose: it
# is Gaza's own southern band, which the model must not see during fitting
# but which is still the develop city. The separate "holdout" part carries
# the whole cities reserved for cross-city generalisation.
DEVELOPMENT_PARTS = ("train", "stack", "val", "test")


def holdout_split_entries():
    """Split entries covering every holdout city at all of its dates.

    Used as the default for CONFIG["split"]["holdout"], so registering a new
    holdout city in CITY_REGISTRY is enough to get it evaluated.
    """
    return [f"{c}@*" for c in held_out_cities()]


def check_split_holdout(split_cfg, parts=DEVELOPMENT_PARTS):
    """Holdout cities that a development split part wrongly mentions.

    Notebook 2's split is hand-written, so this is the cheap guard: pass it
    CONFIG["split"] and it returns [(part, entry, city)] for every entry in
    train/stack/val/test that names a holdout city. An empty list means the
    holdout is intact. The "holdout" part itself is skipped - naming holdout
    cities is exactly what it is for.
    """
    holdout = set(held_out_cities())
    bad = []
    for part, entries in split_cfg.items():
        if part not in parts:
            continue
        for entry in entries:
            city = parse_split_entry(entry)[0]
            if city in holdout:
                bad.append((part, entry, city))
    return bad


# --------------------------------------------------------------------------
# Reloading a finished experiment.
#
# A completed {tag}.pt holds everything needed to score with that model:
# weights, architecture, its params, the channel list, and the mu/sd the
# training data was normalised by. So an evaluation run needs no training
# data at all - which matters because recomputing channel_stats() streams
# every training patch off Drive.
#
# Always take mu/sd from the CHECKPOINT rather than recomputing them. They
# are part of the fitted model: normalising a holdout city by statistics
# recomputed anywhere else silently feeds the network inputs on a different
# scale from the ones it was trained on.
# --------------------------------------------------------------------------

def load_run(models_dir, tag, device="cpu", quiet=True):
    """Rebuild a trained CNN from its checkpoint. No training data needed."""
    path = os.path.join(models_dir, f"{tag}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no checkpoint {path}. Available: {list_saved_runs(models_dir)}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not state.get("complete"):
        raise RuntimeError(
            f"{path} is an unfinished training checkpoint, not a saved run.")
    model = build_model(state["model"], state["channel_names"],
                        device=device, quiet=quiet, **state["params"])
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return {"tag": tag, "model": model, "params": state["params"],
            "history": np.asarray(state["history"], float),
            "best_epoch": int(state["best_epoch"]),
            "val_ap": float(state["val_ap"]),
            "mu": np.asarray(state["mu"]), "sd": np.asarray(state["sd"]),
            "channel_names": list(state["channel_names"]),
            "config": state.get("config")}


def list_saved_runs(models_dir):
    """Tags of every COMPLETED CNN checkpoint in an experiment's models dir."""
    tags = []
    for path in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        if tag.endswith("_training") or tag == "stackers":
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if state.get("complete"):
            tags.append(tag)
    return tags


def list_experiments():
    """Every experiment folder, with what it has finished, newest first."""
    rows = []
    for root in sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*"))):
        if not os.path.isdir(root):
            continue
        models = os.path.join(root, "models")
        metrics = os.path.join(root, "metrics")
        sel_path = os.path.join(metrics, "final_selection.json")
        selection = None
        if os.path.exists(sel_path):
            try:
                with open(sel_path) as fh:
                    sel = json.load(fh)
                selection = f"{sel.get('run')} / {sel.get('variant')}"
            except Exception:
                selection = "unreadable"
        rows.append({
            "experiment": os.path.basename(root),
            "runs": ",".join(list_saved_runs(models)) or "-",
            "stackers": os.path.exists(os.path.join(models, "stackers.pt")),
            "selection": selection or "-",
            "tested": os.path.exists(
                os.path.join(metrics, "final_test_complete.json")),
            "modified": time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(root))),
        })
    return pd.DataFrame(rows).sort_values("modified", ascending=False)


def check_city_windows(city, dates=None):
    """Flag assessment dates whose imagery windows are unusable.

    Returns [(date, reason), ...], empty when every date is fine. See the
    CITY_REGISTRY comment for why these two conditions matter.
    """
    im = PREP["imagery"]
    war = pd.Timestamp(CITY_REGISTRY[city]["war_start"])
    pre_start = war - pd.DateOffset(months=im["pre_months"])
    problems = []
    if pre_start < pd.Timestamp(S1_EARLIEST):
        problems.append((None, f"pre window starts {pre_start.date()}, before "
                               f"Sentinel-1 exists ({S1_EARLIEST})"))
    for d in (dates or CITY_REGISTRY[city]["label_dates"]):
        dt = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        length = pd.DateOffset(months=im["post_months"])
        post_start = dt if im["post_direction"] == "forward" else dt - length
        post_end = dt + length if im["post_direction"] == "forward" else dt
        if post_end < pd.Timestamp(S1_EARLIEST):
            problems.append((d, "no Sentinel-1 data at this date"))
        elif post_start < war:
            problems.append((d, f"post window starts {post_start.date()}, "
                                f"before war_start {war.date()} - the pre-war "
                                f"baseline would postdate the damage"))
    return problems


def list_label_dates(city, geometry=None):
    """All assessment dates for a city that exist in the label folder."""
    geometry = geometry or PREP["label_geometry"]
    pattern = os.path.join(DATA_DIR, f"{city}_*_1_{geometry}.csv")
    return sorted(os.path.basename(p).split("_")[1] for p in glob.glob(pattern))


# --------------------------------------------------------------------------
# Temporal offsets.
#
# The offsets are relative to an assessment date, in days. Offset 0 is the
# labelled date itself and is always present; the others are unlabelled.
# --------------------------------------------------------------------------

def temporal_offsets(include_zero=True):
    """[-24, -12, 0, 12, 24] with the default settings."""
    t = PREP["temporal"]
    if not t["enabled"]:
        return [0] if include_zero else []
    step = int(t["step_days"])
    offs = [-step * i for i in range(1, int(t["n_before"]) + 1)]
    offs += [step * i for i in range(1, int(t["n_after"]) + 1)]
    if include_zero:
        offs.append(0)
    return sorted(offs)


def shift_date(date, days):
    """'20240503', 12 -> '20240515'."""
    d = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}") + pd.Timedelta(days=int(days))
    return d.strftime("%Y%m%d")


def offset_dates(date):
    """{offset in days: calendar date} for one assessment date."""
    return {off: shift_date(date, off) for off in temporal_offsets()}


def temporal_window_days():
    """Length of an offset post window in days, or None to use post_months."""
    return PREP["temporal"]["window_days"]


def post_jobs(city, dates=None):
    """Every post composite a city needs: [(date, window_days, is_labelled)].

    Deduplicated, because two assessment dates can generate the same offset
    date, and ordered so notebook 1 exports the labelled dates first.
    """
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    win = temporal_window_days()
    jobs, seen = [], set()
    for d in dates:                       # labelled dates first, default window
        if (d, None) not in seen:
            seen.add((d, None))
            jobs.append((d, None, True))
    if PREP["temporal"]["enabled"]:
        for d in dates:
            for off, od in offset_dates(d).items():
                if off == 0:
                    continue
                if (od, win) in seen:
                    continue
                seen.add((od, win))
                jobs.append((od, win, False))
    return jobs


# --------------------------------------------------------------------------
# Channel names. Selection everywhere happens by NAME, never by position, so
# a city exported with only SAR and a city with SAR plus optical both work
# without special cases.
#
# The canonical order is ALL PRE CHANNELS THEN ALL POST CHANNELS, because
# that is the order in which the two stored arrays are joined.
# --------------------------------------------------------------------------

S1_CHANNELS = ["s1_pre_VV", "s1_pre_VH", "s1_post_VV", "s1_post_VH"]
S2_CHANNELS = ["s2_pre_B2", "s2_pre_B3", "s2_pre_B4", "s2_pre_B8",
               "s2_post_B2", "s2_post_B3", "s2_post_B4", "s2_post_B8"]

# PWTT columns that could serve as extra tabular features. damage_pts is
# banned on purpose: it counts the UNOSAT points that created the label,
# so using it as a feature would be leakage.
PWTT_BASELINE_COLUMN = "max_change"
PWTT_THRESHOLD = 3.3   # published decision threshold of the PWTT statistic


def stored_channels(phase, sensors=None):
    """Channels held by the pre (or post) array on disk, in stored order."""
    sensors = sensors or PREP["sensors"]
    names = []
    for sensor, chans in [("s1", S1_CHANNELS), ("s2", S2_CHANNELS)]:
        if sensor in sensors:
            names += [c for c in chans if f"_{phase}_" in c]
    return names


def wanted_channels(features, sensors=None):
    """Channel names a feature configuration asks for, in canonical order."""
    names = []
    for phase in ["pre", "post"]:
        if not features.get(f"{phase}_event", True):
            continue
        for name in stored_channels(phase, sensors):
            if name.startswith("s1") and not features.get("sentinel1", True):
                continue
            if name.startswith("s2") and not features.get("sentinel2", False):
                continue
            names.append(name)
    return names


# --------------------------------------------------------------------------
# Label files
# --------------------------------------------------------------------------

def label_path(city, date, geometry=None):
    geometry = geometry or PREP["label_geometry"]
    return os.path.join(DATA_DIR, f"{city}_{date}_1_{geometry}.csv")


def sample_buildings(gdf, n_sample, seed=0):
    """Take a reproducible subset of buildings, identical across dates.

    Plain .sample() picks rows by position, so it only stays consistent if
    every date's file happens to have the same row order. Selecting on the
    building id instead makes the subset identical across dates by
    construction.
    """
    if n_sample is None or len(gdf) <= n_sample:
        return gdf
    ids = np.sort(gdf["system:index"].unique())
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(ids, size=n_sample, replace=False))
    return gdf[gdf["system:index"].isin(keep)]


def load_labels(city, date, geometry=None):
    """Read one PWTT CSV into a GeoDataFrame with centroid points."""
    path = label_path(city, date, geometry)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Check that the label folder is on Drive "
            f"and that {city} has an assessment on {date}.")
    df = pd.read_csv(path, low_memory=False)
    geoms = [shape(json.loads(g)) for g in df[".geo"]]
    gdf = gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry=geoms, crs="EPSG:4326")
    # centroids are computed in a projected system, then converted back
    gdf["centroid"] = gdf.geometry.to_crs(3857).centroid.to_crs(4326)
    return gdf


# --------------------------------------------------------------------------
# File names. The settings that shaped a file are baked into its name, so
# changing a setting forces a new file instead of silently reusing an old one.
# --------------------------------------------------------------------------

def prep_tag():
    """Identifies the patch geometry and the building sample."""
    return (f"p{PREP['patch_size']}_{PREP['label_geometry']}"
            f"_n{PREP['n_sample']}_{''.join(PREP['sensors'])}")


def _pre_tag():
    return f"pre{PREP['imagery']['pre_months']}m"


def _post_tag(window_days=None):
    im = PREP["imagery"]
    w = f"{int(window_days)}d" if window_days else f"{im['post_months']}m"
    return f"post_{im['post_direction']}{w}"


def pre_raster_path(city, sensor):
    """The pre-war composite. One per city: the window never moves."""
    os.makedirs(RASTER_DIR, exist_ok=True)
    return os.path.join(RASTER_DIR, f"{city.lower()}_{sensor}_{_pre_tag()}.tif")


def post_raster_path(city, date, sensor, window_days=None):
    """One post composite per date. Non-default windows get their own name."""
    os.makedirs(RASTER_DIR, exist_ok=True)
    return os.path.join(
        RASTER_DIR, f"{city.lower()}_{date}_{sensor}_{_post_tag(window_days)}.tif")


def export_meta_path(city):
    """Where the export decisions shared by every raster of a city are kept.

    Two things have to be decided ONCE per city and then reused:

    orbit - pre and post must come from the same relative orbit. A different
        orbit means a different viewing geometry, and the backscatter step
        that produces looks exactly like damage. When pre and post lived in
        one export the orbit was chosen per file; now that they are separate
        files it has to be pinned for the city.
    bbox - every raster must cover the same ground, or a building sits inside
        one date's raster and off the edge of another's.
    """
    os.makedirs(RASTER_DIR, exist_ok=True)
    return os.path.join(RASTER_DIR, f"{city.lower()}_export.json")


def read_export_meta(city):
    path = export_meta_path(city)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def write_export_meta(city, **kwargs):
    meta = read_export_meta(city)
    meta.update(kwargs)
    with open(export_meta_path(city), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def buildings_path(city):
    """The canonical building table every patch array is aligned to."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    return os.path.join(PROCESSED_DIR, f"{city.lower()}_{prep_tag()}_buildings.parquet")


def pre_patch_paths(city):
    """(patches, valid mask) for the pre-war channels of a city."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stem = os.path.join(PROCESSED_DIR, f"{city.lower()}_{prep_tag()}_{_pre_tag()}")
    return stem + "_X.npy", stem + "_valid.npy"


def post_patch_paths(city, date, window_days=None):
    """(patches, valid mask) for the post channels of one city and date."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stem = os.path.join(
        PROCESSED_DIR,
        f"{city.lower()}_{date}_{prep_tag()}_{_post_tag(window_days)}")
    return stem + "_X.npy", stem + "_valid.npy"


# --------------------------------------------------------------------------
# Writing patches.
#
# Patches go to an uncompressed .npy as float16. float16 because backscatter
# in dB spans about -35 to +5, where it resolves ~0.01 dB, far finer than
# radar speckle. Uncompressed because a .npz is a zip archive and a
# compressed member has no fixed position on disk, so it cannot be
# memory-mapped: reading one patch would mean decompressing the whole array.
#
# open_patch_writer hands back a memory-mapped array to fill row by row, so
# notebook 1 never holds the full stack in RAM.
# --------------------------------------------------------------------------

def open_patch_writer(path, n_rows, n_channels, patch=None):
    from numpy.lib.format import open_memmap
    patch = patch or PREP["patch_size"]
    return open_memmap(path, mode="w+", dtype=np.float16,
                       shape=(n_rows, n_channels, patch, patch))


# --------------------------------------------------------------------------
# Reading a city back.
#
# X arrays come back as float16 memory maps: the pixels stay on disk and the
# operating system pages in only what is actually read, so several gigabytes
# of patches cost almost no memory. Never call np.asarray on a whole one.
# --------------------------------------------------------------------------

def load_city(city, dates=None):
    """Canonical table plus the memory-mapped pre patches for one city."""
    tab_path = buildings_path(city)
    pre_x, pre_v = pre_patch_paths(city)
    if not os.path.exists(pre_x):
        raise FileNotFoundError(
            f"{pre_x} not found. Run notebook 1 with the current PREP "
            f"settings first. (Files from the old one-array-per-date layout "
            f"are not readable here - see MIGRATION in pipeline.py.)")
    table = gpd.read_parquet(tab_path)
    d = {
        "city": city,
        "table": table,
        "dates": list(dates or CITY_REGISTRY[city]["label_dates"]),
        "lat": table["lat"].to_numpy(float),
        "lon": table["lon"].to_numpy(float),
        "xy": np.c_[table["x_m"].to_numpy(float), table["y_m"].to_numpy(float)],
        "pre": np.load(pre_x, mmap_mode="r"),
        "pre_valid": np.load(pre_v),
        "pre_channel_names": stored_channels("pre"),
        "post_channel_names": stored_channels("post"),
        "_post": {},
    }
    return d


def post_arrays(d, date, window_days="auto"):
    """(patches, valid) for one post date, opened once and cached."""
    if window_days == "auto":
        window_days = None if date in d["dates"] else temporal_window_days()
    key = (date, window_days)
    if key not in d["_post"]:
        xp, vp = post_patch_paths(d["city"], date, window_days)
        if not os.path.exists(xp):
            raise FileNotFoundError(
                f"{xp} not found. Notebook 1 has not cut the post patches for "
                f"{d['city']} {date}. If this is a temporal offset date, check "
                f"that PREP['temporal'] matches the settings used there.")
        d["_post"][key] = (np.load(xp, mmap_mode="r"), np.load(vp))
    return d["_post"][key]


def usable_mask(d, date, window_days="auto"):
    """Rows whose patch is complete in BOTH the pre and this post array."""
    return d["pre_valid"] & post_arrays(d, date, window_days)[1]


def city_source(d, date, features, rows=None, window_days="auto"):
    """A PatchSource joining pre and post channels for the given rows.

    rows: global positions in the canonical table. They are filtered to the
    rows that are actually usable at this date, and the surviving positions
    are returned alongside so the caller can line scores back up.

    Returns (source, rows_kept).
    """
    post, _ = post_arrays(d, date, window_days)
    ok = usable_mask(d, date, window_days)
    rows = np.arange(len(ok)) if rows is None else np.asarray(rows)
    if rows.dtype == bool:
        rows = np.where(rows)[0]
    rows = rows[ok[rows]]

    want = wanted_channels(features)
    pieces = []
    for arr, names in [(d["pre"], d["pre_channel_names"]),
                       (post, d["post_channel_names"])]:
        idx = [names.index(n) for n in want if n in names]
        if idx:
            pieces.append((arr, idx))
    missing = [n for n in want
               if n not in d["pre_channel_names"] + d["post_channel_names"]]
    if missing:
        raise ValueError(
            f"{d['city']} was preprocessed with "
            f"{d['pre_channel_names'] + d['post_channel_names']} but the "
            f"experiment asks for {missing}. Add the sensor to "
            f"PREP['sensors'] and re-run notebook 1.")
    return PatchSource.from_arrays(pieces, rows), rows


def summarize_processed(city, dates=None):
    """Per-date row counts and damaged share, without touching any pixels."""
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    table = gpd.read_parquet(buildings_path(city))
    pre_valid = np.load(pre_patch_paths(city)[1])
    rows = []
    for dt in dates:
        _, vp = post_patch_paths(city, dt)
        ok = pre_valid & np.load(vp)
        y = table[f"class_{dt}"].to_numpy(int)[ok]
        rows.append({"date": dt, "buildings": int(ok.sum()),
                     "damaged": int(y.sum()),
                     "damaged %": round(float(y.mean()) * 100, 2)})
    return pd.DataFrame(rows)


def inventory(city, dates=None, sensors=None):
    """What already exists on disk for one city, per raster and patch array.

    Everything downstream skips work whose output file is already there, so
    nothing here is required for correctness - it exists so you can SEE, before
    starting a run, that Gaza's 16 composites are already on Drive and only the
    new cities will actually hit Earth Engine.

    Returns a DataFrame with one row per (date, sensor) plus the pre row, and
    columns saying whether the raster, the patch array and its valid mask are
    present. Reads no pixels: this is os.path.exists and file sizes only.
    """
    sensors = sensors or PREP["sensors"]
    dates = dates or CITY_REGISTRY[city]["label_dates"]

    def _row(kind, date, window_days, sensor, raster, x_path, v_path):
        have_r = os.path.exists(raster)
        have_x = os.path.exists(x_path)
        return {
            "city": city, "kind": kind, "date": date or "-", "sensor": sensor,
            "raster": have_r,
            "raster_MB": round(os.path.getsize(raster) / 1e6, 1) if have_r else 0.0,
            "patches": have_x,
            "patch_GB": round(os.path.getsize(x_path) / 1e9, 2) if have_x else 0.0,
            "valid_mask": os.path.exists(v_path),
        }

    rows = []
    px, pv = pre_patch_paths(city)
    for s in sensors:
        rows.append(_row("pre", None, None, s, pre_raster_path(city, s), px, pv))
    for date, window_days, is_labelled in post_jobs(city, dates):
        xp, vp = post_patch_paths(city, date, window_days)
        for s in sensors:
            rows.append(_row("labelled" if is_labelled else "offset", date,
                             window_days, s,
                             post_raster_path(city, date, s, window_days),
                             xp, vp))
    return pd.DataFrame(rows)


def inventory_summary(cities=None, dates_by_city=None):
    """One row per city: how much is done, how much is still to download."""
    cities = cities or list(CITY_REGISTRY)
    rows = []
    for city in cities:
        dates = (dates_by_city or {}).get(city)
        inv = inventory(city, dates)
        rows.append({
            "city": city,
            "role": city_role(city),
            "dates": len(dates or CITY_REGISTRY[city]["label_dates"]),
            "rasters": f"{int(inv['raster'].sum())}/{len(inv)}",
            "patches": f"{int(inv['patches'].sum())}/{len(inv)}",
            "on_disk_GB": round(inv["raster_MB"].sum() / 1e3
                                + inv["patch_GB"].sum(), 2),
            "to_export": int((~inv["raster"]).sum()),
            "to_cut": int((~inv["patches"]).sum()),
            "table": os.path.exists(buildings_path(city)),
        })
    return pd.DataFrame(rows)


def cut_patches(city, date, lonlat, patch=None, window_days=None):
    """Read patches straight from the rasters. Returns (X, keep_mask).

    Far slower than the stored patches, so this is for one-off inference or
    for re-cutting at a different patch_size without touching Earth Engine -
    never for the training loop.
    """
    patch = patch or PREP["patch_size"]
    half = patch // 2
    lonlat = np.asarray(lonlat, dtype=float)
    n = len(lonlat)
    blocks, keep = [], np.ones(n, dtype=bool)

    sources = ([pre_raster_path(city, s) for s in PREP["sensors"]] +
               [post_raster_path(city, date, s, window_days) for s in PREP["sensors"]])
    for path in sources:
        with rasterio.open(path) as src:
            inv = ~src.transform
            cols, rows = inv * (lonlat[:, 0], lonlat[:, 1])
            r0 = np.round(rows).astype(int) - half
            c0 = np.round(cols).astype(int) - half
            inside = ((r0 >= 0) & (c0 >= 0) &
                      (r0 + patch <= src.height) & (c0 + patch <= src.width))
            out = np.full((n, src.count, patch, patch), np.nan, np.float32)
            for i in np.argsort(r0):          # ascending rows = fewer block reads
                if inside[i]:
                    out[i] = src.read(window=Window(int(c0[i]), int(r0[i]),
                                                    patch, patch))
            keep &= inside
        blocks.append(out)

    X = np.concatenate(blocks, axis=1)
    keep &= np.isfinite(X).all(axis=(1, 2, 3))   # radar nodata is -inf, not NaN
    return X[keep], keep


# --------------------------------------------------------------------------
# Lazy patch access.
#
# Notebook 2 needs to (a) keep only some channels, (b) keep only the rows in
# one latitude band, (c) glue several dates together, and (d) join a pre
# array to a post array. Doing any of those with ordinary numpy indexing
# copies the whole array into memory, which undoes the memory mapping and is
# what runs Colab out of RAM.
#
# PatchSource records those operations instead of performing them, and reads
# pixels only when a batch is actually asked for. A segment is
#
#     ([(array, channel indices), ...], row indices)
#
# where the pieces share one row index and are concatenated along the
# channel axis - that is what makes the pre/post join free.
# --------------------------------------------------------------------------

class PatchSource:
    """A lazy view over one or more patch arrays on disk."""

    def __init__(self, segments):
        self.segments = [([(a, list(c)) for a, c in pieces],
                          np.asarray(r, dtype=np.int64))
                         for pieces, r in segments]
        if not self.segments:
            raise ValueError("a PatchSource needs at least one segment")
        lengths = [len(r) for _, r in self.segments]
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        pieces0, _ = self.segments[0]
        n_ch = sum(len(c) for _, c in pieces0)
        self.shape = (int(self.offsets[-1]), n_ch) + tuple(pieces0[0][0].shape[2:])
        self.dtype = pieces0[0][0].dtype

    @classmethod
    def from_arrays(cls, pieces, rows=None):
        """pieces: [(array, channel indices), ...] sharing one row index."""
        if rows is None:
            rows = np.arange(len(pieces[0][0]))
        return cls([(pieces, rows)])

    @classmethod
    def from_array(cls, array, channels=None, rows=None):
        ch = list(range(array.shape[1])) if channels is None else list(channels)
        return cls.from_arrays([(array, ch)], rows)

    def rows(self, mask_or_idx):
        """A new source keeping only these rows (boolean mask or indices)."""
        idx = np.asarray(mask_or_idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        segs = []
        for s, (pieces, rws) in enumerate(self.segments):
            lo, hi = self.offsets[s], self.offsets[s + 1]
            take = idx[(idx >= lo) & (idx < hi)] - lo
            if len(take):
                segs.append((pieces, rws[take]))
        return PatchSource(segs)

    @staticmethod
    def concat(sources):
        """Stack several sources end to end without reading any pixels."""
        return PatchSource([seg for src in sources for seg in src.segments])

    def __len__(self):
        return self.shape[0]

    def _read(self, gidx):
        out = np.empty((len(gidx),) + self.shape[1:], self.dtype)
        seg_of = np.searchsorted(self.offsets, gidx, side="right") - 1
        for s in np.unique(seg_of):
            here = seg_of == s
            pieces, rws = self.segments[s]
            local = rws[gidx[here] - self.offsets[s]]
            # memory maps are much faster read in ascending order, so sort,
            # read, then put the rows back where the caller expects them
            order = np.argsort(local)
            wanted = local[order]
            block = np.concatenate(
                [np.asarray(arr[wanted])[:, ch] for arr, ch in pieces], axis=1)
            slot = np.empty_like(block)
            slot[order] = block
            out[here] = slot
        return out

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._read(np.array([key], dtype=np.int64))[0]
        if isinstance(key, slice):
            return self._read(np.arange(*key.indices(len(self)), dtype=np.int64))
        key = np.asarray(key)
        if key.dtype == bool:
            key = np.where(key)[0]
        return self._read(key.astype(np.int64))

    def channel_stats(self, batch=4096):
        """Per-channel mean and std in float32, from one streamed pass.

        Accumulating in float64 matters: summing millions of float16 values
        in float16 builds up serious rounding error.
        """
        total = np.zeros(self.shape[1], np.float64)
        total_sq = np.zeros(self.shape[1], np.float64)
        px = 0
        for i in range(0, len(self), batch):
            b = self[i:i + batch].astype(np.float32)
            total += b.sum(axis=(0, 2, 3))
            total_sq += (b.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
            px += b.shape[0] * b.shape[2] * b.shape[3]
        mu = total / px
        var = np.maximum(total_sq / px - mu ** 2, 0)
        shape = (1, self.shape[1], 1, 1)
        return (mu.astype(np.float32).reshape(shape),
                (np.sqrt(var).astype(np.float32) + 1e-6).reshape(shape))


# --------------------------------------------------------------------------
# Train / validation / test splits.
#
# A split entry is either "City" (the whole city) or "City:lo-hi" where lo
# and hi are QUANTILES of building latitude, optionally followed by "@date".
# Bands carve one city into spatially disjoint regions: a random split would
# leak information through spatial autocorrelation (neighbouring buildings
# look alike and share their fate), a latitude band split cannot.
# --------------------------------------------------------------------------

def parse_split_entry(entry):
    base = entry.split("@")[0]
    if ":" in base:
        city, band = base.split(":")
        lo, hi = (float(v) for v in band.split("-"))
        return city, lo, hi
    return base, 0.0, 1.0


def parse_entry(entry):
    """'Gaza:0.00-0.33@20240906' -> ('Gaza', 0.0, 0.33, '20240906')."""
    base, _, date = entry.partition("@")
    city, lo, hi = parse_split_entry(base)
    return city, lo, hi, (date or None)


def expand_entry(entry, dates=None):
    """'Gaza:0.33-0.42@*' -> one entry per registered assessment date.

    Multi-date validation and stacking sets are the point of this: a
    threshold fitted at one date is calibrated to that date's prevalence,
    and Gaza's prevalence climbs from 54 to 65 percent across the three
    assessments, so a single-date threshold is systematically wrong for the
    others.
    """
    city, lo, hi, date = parse_entry(entry)
    if date is None or date != "*":
        return [entry]
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    base = entry.split("@")[0]
    return [f"{base}@{d}" for d in dates]


def expand_split(split_cfg):
    """Resolve every '@*' in a split configuration."""
    return {part: [e for entry in entries for e in expand_entry(entry)]
            for part, entries in split_cfg.items()}


def band_mask(lat, lo, hi):
    """Boolean mask selecting the latitude band between two quantiles."""
    qlo, qhi = np.quantile(lat, [lo, hi])
    return (lat >= qlo) & (lat <= qhi)


def split_assignment(lat, city, split_cfg):
    """Label every building of a city as train / val / test / unused."""
    role = np.array(["unused"] * len(lat), dtype=object)
    for part in split_cfg:
        for entry in split_cfg[part]:
            c, lo, hi = parse_split_entry(entry)
            if c == city:
                role[band_mask(lat, lo, hi)] = part
    return role


def strip_dates(split_cfg):
    """The same split without @date suffixes, for functions that map bands."""
    return {k: sorted({e.split("@")[0] for e in v}) for k, v in split_cfg.items()}


# --------------------------------------------------------------------------
# Augmentation 1: temporal label propagation.
#
# Uses the physics of destruction: a building marked damaged at one
# assessment date stays damaged at every later date, and strictly before its
# first damaged label it counts as intact. With the canonical table this is
# a cumulative maximum over the date axis rather than an id join.
# --------------------------------------------------------------------------

def label_matrix(table, dates):
    """(n buildings, n dates) integer label matrix, oldest date first."""
    return np.stack([table[f"class_{d}"].to_numpy(int) for d in dates], axis=1)


def propagate_labels(Y):
    """Once damaged, damaged at every later date. Y is (n, n_dates)."""
    return np.maximum.accumulate(Y, axis=1)


# --------------------------------------------------------------------------
# Augmentation 2: spatial smoothing of prediction scores.
#
# Damage is spatially clustered (shelling destroys blocks, not isolated
# houses), so at prediction time each building's score is blended with the
# average score of its k nearest neighbours. weight 0 returns the original
# scores, weight 1 pure neighbourhood consensus.
#
# The neighbour search must stay INSIDE one evaluation region: smoothing
# across the train/test boundary would let training-area scores influence
# test predictions, which is exactly the leak the latitude bands prevent.
# --------------------------------------------------------------------------

def spatial_smooth(xy, scores, k=8, weight=0.3):
    """Blend scores with the mean of the k nearest neighbours' scores.

    xy: (n, 2) coordinates in metres, so distances are physical.
    """
    from sklearn.neighbors import BallTree
    if len(scores) < 2:
        return scores
    tree = BallTree(xy)
    k_eff = min(k + 1, len(scores))          # +1 because the nearest
    _, nn = tree.query(xy, k=k_eff)          # neighbour of a point is itself
    neighbour_mean = scores[nn[:, 1:]].mean(axis=1)
    return (1 - weight) * scores + weight * neighbour_mean


# --------------------------------------------------------------------------
# Neighbourhood features for a second-stage model.
#
# spatial_smooth above blends a score with its neighbours using a weight we
# fix by hand. These features hand the same information to a model and let it
# work out the weighting itself, per building rather than globally: a score of
# 0.6 surrounded by 0.9s means something different from a 0.6 surrounded by
# 0.1s, and the right correction is not the same in both cases.
# --------------------------------------------------------------------------

def neighbour_features(xy, scores, ks=(8, 32)):
    """Features describing a building's score against its neighbours'.

    Neighbours must come from the same evaluation region: computing them
    across a train/test boundary would let training-area scores leak into
    test predictions, which is exactly what the latitude bands prevent.
    """
    from sklearn.neighbors import BallTree

    n = len(scores)
    feats = {"score": scores.astype(np.float32)}
    if n < 2:
        for k in ks:
            for suffix in ["mean", "std", "max", "min", "dev", "dist"]:
                feats[f"k{k}_{suffix}"] = np.zeros(n, np.float32)
        return pd.DataFrame(feats)

    tree = BallTree(xy)
    max_k = min(max(ks) + 1, n)
    dist, nn = tree.query(xy, k=max_k)      # column 0 is the point itself

    for k in ks:
        k_eff = min(k, max_k - 1)
        nb = scores[nn[:, 1:k_eff + 1]]
        d = dist[:, 1:k_eff + 1]
        feats[f"k{k}_mean"] = nb.mean(axis=1)
        feats[f"k{k}_std"] = nb.std(axis=1)        # the spatial spread of scores
        feats[f"k{k}_max"] = nb.max(axis=1)
        feats[f"k{k}_min"] = nb.min(axis=1)
        feats[f"k{k}_dev"] = scores - nb.mean(axis=1)   # how far the building
        feats[f"k{k}_dist"] = d.mean(axis=1)            # stands out locally

    # the nearest few scores individually, sorted so the model sees "the
    # strongest neighbour, the second strongest..." rather than an arbitrary order
    k_near = min(8, max_k - 1)
    near = np.sort(scores[nn[:, 1:k_near + 1]], axis=1)[:, ::-1]
    for i in range(k_near):
        feats[f"nb{i + 1}"] = near[:, i]

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in feats.items()})


# --------------------------------------------------------------------------
# Temporal features for the second-stage model.
#
# The CNN is run over the SAME buildings using post imagery from a few dates
# either side of the assessment, against the same pre-war baseline, so the
# scores are directly comparable. What separates real damage from speckle is
# its shape in time: destruction appears once and persists, speckle flickers.
#
# NaN is used deliberately for offsets where a building's patch is missing.
# XGBoost handles missing values natively by learning a default direction per
# split, which is a better answer than imputing a score that was never
# measured.
# --------------------------------------------------------------------------

def temporal_features(scores_by_offset, prefix="t"):
    """Turn {offset in days: score array} into a feature table.

    Offset 0 is the labelled date. Every array must be the same length and
    aligned to the same buildings.
    """
    offs = sorted(scores_by_offset)
    if 0 not in offs:
        raise ValueError("temporal_features needs the labelled date at offset 0")
    S = np.stack([np.asarray(scores_by_offset[o], np.float32) for o in offs], axis=1)
    t = np.asarray(offs, np.float32)
    s0 = S[:, offs.index(0)]

    feats = {}
    for j, o in enumerate(offs):
        tag = f"{prefix}{o:+d}".replace("+0", "0")
        if o != 0:
            feats[tag] = S[:, j]
            feats[f"{tag}_dev"] = s0 - S[:, j]   # change relative to the label date

    past, future = t < 0, t > 0
    with np.errstate(invalid="ignore"):
        if past.any():
            feats[f"{prefix}_past_mean"] = np.nanmean(S[:, past], axis=1)
            feats[f"{prefix}_past_max"] = np.nanmax(S[:, past], axis=1)
            feats[f"{prefix}_dev_past"] = s0 - feats[f"{prefix}_past_mean"]
        if future.any():
            feats[f"{prefix}_future_mean"] = np.nanmean(S[:, future], axis=1)
            feats[f"{prefix}_future_min"] = np.nanmin(S[:, future], axis=1)
            feats[f"{prefix}_dev_future"] = feats[f"{prefix}_future_mean"] - s0
        feats[f"{prefix}_mean"] = np.nanmean(S, axis=1)
        feats[f"{prefix}_std"] = np.nanstd(S, axis=1)       # flicker vs persistence
        feats[f"{prefix}_min"] = np.nanmin(S, axis=1)
        feats[f"{prefix}_max"] = np.nanmax(S, axis=1)
        feats[f"{prefix}_range"] = feats[f"{prefix}_max"] - feats[f"{prefix}_min"]
        feats[f"{prefix}_n_obs"] = np.isfinite(S).sum(axis=1).astype(np.float32)

        # least squares slope per building, computed on the observed points
        # only. A rising series is what genuine, persisting damage looks like.
        ok = np.isfinite(S)
        w = ok.astype(np.float32)
        n = w.sum(axis=1)
        Sz = np.where(ok, S, 0.0)
        tm = (w * t).sum(axis=1) / np.maximum(n, 1)
        sm = Sz.sum(axis=1) / np.maximum(n, 1)
        cov = (w * (t - tm[:, None]) * (Sz - sm[:, None] * w)).sum(axis=1)
        var = (w * (t - tm[:, None]) ** 2).sum(axis=1)
        slope = np.where((n >= 2) & (var > 0), cov / np.maximum(var, 1e-9), np.nan)
        feats[f"{prefix}_slope"] = slope.astype(np.float32)

        # how monotone the series is: 1.0 means every step was non-decreasing
        steps = np.diff(S, axis=1)
        good = np.isfinite(steps)
        rising = (good & (np.nan_to_num(steps, nan=-1.0) >= 0)).sum(axis=1)
        feats[f"{prefix}_monotone"] = np.where(
            good.sum(axis=1) > 0, rising / np.maximum(good.sum(axis=1), 1),
            np.nan).astype(np.float32)

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in feats.items()})


# --------------------------------------------------------------------------
# Models. A registry keyed by name: a new architecture is a class with
# forward(x) plus one decorator line, and it becomes selectable through
# CONFIG["model"] in notebook 2. Every model takes a (batch, channels,
# patch, patch) tensor and returns one logit per patch.
#
# Every constructor argument beyond n_channels is a hyperparameter Optuna can
# search, which is why width, depth and dropout are arguments rather than
# constants.
# --------------------------------------------------------------------------

MODEL_REGISTRY = {}


def register_model(name):
    def wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrap


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(), nn.MaxPool2d(2))


@register_model("small_cnn")
class SmallCNN(nn.Module):
    """The from-scratch baseline. All channels stacked."""

    def __init__(self, n_channels, width=16, depth=3, dropout=0.3):
        super().__init__()
        chans = [n_channels] + [width * (2 ** i) for i in range(depth)]
        self.features = nn.Sequential(
            *[conv_block(chans[i], chans[i + 1]) for i in range(depth)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(chans[-1], 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


@register_model("siamese")
class SiameseCNN(nn.Module):
    """One shared encoder applied to pre and post imagery separately.

    The classifier head sees pre features, post features and their
    difference. The explicit difference is the inductive bias that plain
    channel stacking would have to discover on its own. Needs both
    pre_event and post_event enabled.
    """

    def __init__(self, n_channels, width=32, depth=3, dropout=0.4,
                 head_dim=128, head_dropout=0.2):
        super().__init__()
        assert n_channels % 2 == 0, "siamese needs matching pre and post channels"
        chans = [n_channels // 2] + [width * (2 ** i) for i in range(depth)]
        self.encoder = nn.Sequential(
            *[conv_block(chans[i], chans[i + 1]) for i in range(depth)],
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        feat = chans[-1]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * feat, head_dim),
                                  nn.ReLU(), nn.Dropout(head_dropout),
                                  nn.Linear(head_dim, 1))

    def set_channel_split(self, names):
        """Resolve which channel indices are pre and which are post."""
        self.pre_idx = [i for i, n in enumerate(names) if "_pre_" in n]
        self.post_idx = [i for i, n in enumerate(names) if "_post_" in n]
        assert len(self.pre_idx) == len(self.post_idx)

    def forward(self, x):
        f_pre = self.encoder(x[:, self.pre_idx])
        f_post = self.encoder(x[:, self.post_idx])
        f = torch.cat([f_pre, f_post, f_post - f_pre], dim=1)
        return self.head(f).squeeze(1)

class SpatialDropout(nn.Module):
    """Dropout that removes whole feature maps rather than single pixels.

    Ordinary dropout does almost nothing inside a conv stack: neighbouring
    pixels in a feature map are strongly correlated, so zeroing one leaves
    its neighbours to carry the same information straight through. Dropping
    the entire channel forces the next layer to cope without that feature.
    """

    def __init__(self, p):
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        return self.drop(x)


@register_model("tiny_cnn")
class TinyCNN(nn.Module):
    """A deliberately small CNN, regularised where it actually bites.

    Three changes from SmallCNN, each aimed at a specific failure seen in the
    siamese and small_cnn runs:

    1. Dropout lives INSIDE the conv stack, as channel dropout. SmallCNN puts
       one Dropout before a Linear(chans[-1], 1) - at depth 2 that is dropout
       on a 32-vector feeding 33 parameters, which is why raising it to 0.6
       changed almost nothing.

    2. The head keeps the centre. SmallCNN global-average-pools the final map,
       so a change at the patch edge and the same change dead centre produce
       identical features. The patch is 320 m across and centred on ONE
       building, so that hands neighbourhood inference to the CNN by accident
       - which is stage two's job. Here the centre cell is concatenated with
       the global average, and the model can weigh them.

    3. Channels grow more slowly (`growth`, default 1.5 rather than 2), so
       depth can be increased for receptive field without the parameter count
       exploding.

    At the defaults this is roughly 3k parameters against ~350k patches. Note
    the effective sample is far smaller than that: a 32 px patch is 320 m
    across, so neighbouring buildings share pixels and the training band holds
    a few thousand independent neighbourhoods, not 350k. That mismatch, not
    raw capacity, is what drove the overfitting.
    """

    def __init__(self, n_channels, width=8, depth=3, dropout=0.15,
                 growth=1.5, head_dropout=0.2, use_centre=True):
        super().__init__()
        chans = [n_channels] + [max(4, int(round(width * growth ** i)))
                                for i in range(depth)]
        blocks = []
        for i in range(depth):
            blocks.append(conv_block(chans[i], chans[i + 1]))
            if dropout > 0:
                blocks.append(SpatialDropout(dropout))
        self.features = nn.Sequential(*blocks)

        self.use_centre = use_centre
        feat = chans[-1] * (2 if use_centre else 1)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(feat, 1))

    def forward(self, x):
        f = self.features(x)
        pooled = f.mean(dim=(2, 3))
        if self.use_centre:
            # the target building sits at the patch centre by construction
            i = f.shape[-2] // 2
            j = f.shape[-1] // 2
            pooled = torch.cat([pooled, f[:, :, i, j]], dim=1)
        return self.head(pooled).squeeze(1)


def build_model(name, channel_names, device="cpu", quiet=False, **hparams):
    """Create a model from the registry, configured for the active channels.

    Unknown keyword arguments are dropped rather than raising, so a single
    Optuna parameter dictionary can be passed to any architecture.
    """
    cls = MODEL_REGISTRY[name]
    accepted = set(inspect.signature(cls.__init__).parameters)
    kwargs = {k: v for k, v in hparams.items() if k in accepted}
    model = cls(n_channels=len(channel_names), **kwargs)
    if hasattr(model, "set_channel_split"):
        model.set_channel_split(channel_names)
    if not quiet:
        n = sum(p.numel() for p in model.parameters())
        extra = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        print(f"model {name}: {n:,} parameters" + (f"  ({extra})" if extra else ""))
    return model.to(device)


@torch.no_grad()
def predict_probs(model, X, mu=None, sd=None, device=None, batch=256,
                  zero_channels=None):
    """Damage probabilities for a PatchSource or a plain array.

    Normalization happens here, one batch at a time, so a full-precision copy
    of the data never exists. This is what keeps scoring the temporal offsets
    cheap: five extra passes over memory-mapped patches, no extra RAM.
    """
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = np.asarray(X[i:i + batch]).astype(np.float32)
        if mu is not None:
            xb = (xb - mu) / sd
        if zero_channels:
            xb[:, list(zero_channels)] = 0.0
        out.append(torch.sigmoid(model(torch.from_numpy(xb).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def best_f1_threshold(y_true, scores):
    """The threshold maximising F1, and that F1. Fit on VALIDATION only."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    if len(thr) == 0:
        return 0.5, 0.0
    j = int(np.argmax(f1))
    return float(thr[j]), float(f1[j])


# --------------------------------------------------------------------------
# Experiment folders
# --------------------------------------------------------------------------

def experiment_dirs(experiment_name):
    """Create experiments/{name}/{models,metrics,figures,predictions}."""
    root = os.path.join(EXPERIMENTS_DIR, experiment_name)
    dirs = {name: os.path.join(root, name)
            for name in ["models", "metrics", "figures", "predictions"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return root, dirs


# --------------------------------------------------------------------------
# MIGRATION from the old layout
#
# The previous version stored one 4-band raster and one 4-channel patch array
# per city and date. The rasters can be reused: the pre bands of any of them
# are the city's pre-war composite, and the post bands are that date's post
# composite. split_legacy_rasters() does that locally, so the only exports
# you actually have to run are the new temporal offset dates.
#
# The old PATCH arrays are not reusable, because the canonical row order
# changed. Re-run section 3 of notebook 1 to rebuild them.
# --------------------------------------------------------------------------

def legacy_raster_path(city, date, sensor):
    im = PREP["imagery"]
    tag = f"pre{im['pre_months']}m_{im['post_direction']}{im['post_months']}m"
    return os.path.join(RASTER_DIR, f"{city.lower()}_{date}_{sensor}_{tag}.tif")


def split_legacy_rasters(city, dates=None, sensors=None, keep=True):
    """Split old combined pre+post GeoTIFFs into the new separate files.

    Saves re-downloading everything from Earth Engine. The pre composite is
    written once (from the first date found) and the post composite once per
    date. Existing new-layout files are never overwritten.
    """
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    sensors = sensors or PREP["sensors"]
    n_pre = {s: False for s in sensors}
    made = []
    for sensor in sensors:
        for date in dates:
            src_path = legacy_raster_path(city, date, sensor)
            if not os.path.exists(src_path):
                continue
            with rasterio.open(src_path) as src:
                names = list(src.descriptions)
                pre_idx = [i + 1 for i, n in enumerate(names) if n and "pre" in n]
                post_idx = [i + 1 for i, n in enumerate(names) if n and "post" in n]
                if not pre_idx or not post_idx:      # fall back to halves
                    half = src.count // 2
                    pre_idx = list(range(1, half + 1))
                    post_idx = list(range(half + 1, src.count + 1))

                def write(out_path, idx):
                    profile = src.profile.copy()
                    profile.update(count=len(idx))
                    with rasterio.open(out_path, "w", **profile) as dst:
                        for k, b in enumerate(idx, start=1):
                            dst.write(src.read(b), k)
                            if names[b - 1]:
                                dst.set_band_description(k, names[b - 1])
                    made.append(out_path)

                out_pre = pre_raster_path(city, sensor)
                if not n_pre[sensor] and not os.path.exists(out_pre):
                    write(out_pre, pre_idx)
                n_pre[sensor] = True
                out_post = post_raster_path(city, date, sensor)
                if not os.path.exists(out_post):
                    write(out_post, post_idx)
    for p in made:
        print(f"  wrote {os.path.basename(p)}")
    if not made:
        print("  nothing to split (either no legacy rasters, or already done)")
    return made
