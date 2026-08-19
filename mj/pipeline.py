"""
Shared code for the war damage detection project.

All three notebooks import this file, so the city registry, file paths, data
loading, the models and the two augmentation helpers exist in exactly one
place. Upload pipeline.py to the project folder on Google Drive.

Expected folder layout on Google Drive (everything lives under BASE):

    one_month/      PWTT label files, e.g. Gaza_20240503_1_footprints.csv
    pipeline.py     this file
    rasters/        Sentinel exports        (written once by notebook 1)
    processed/      patch arrays per city   (written once by notebook 1)
    experiments/    one folder per run      (written by notebook 2)

The division of labour between the notebooks:

    1_preprocessing_eda    downloads imagery ONCE, cuts patches, saves them,
                           and explores the labels and the imagery
    2_training_evaluation  trains a model on the saved patches and evaluates
                           it against the PWTT baseline
    3_interactive_map      applies a trained model and shows the predictions
                           on an interactive map
"""

import os
import glob
import json

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

    # cap on buildings per city and date, None keeps every row
    "n_sample": None, #4000,
    "seed": 0,

    "imagery": {
        "gee_project": "test-cnn-war-damage",  # your Earth Engine project id
        "pre_months": 12,             # pre war baseline window length
        "post_months": 1,             # post window length, matches "one_month"
        # "forward" is the PWTT convention
        "post_direction": "forward",
        "scale": 10,                  # metres per pixel
        "aoi_buffer_deg": 0.005,      # padding so edge buildings get a patch
        "s2_max_cloud": 60,           # max cloud percentage for Sentinel 2
        "min_aoi_coverage": 0.90,     # warn if the chosen orbit misses the city
    },
}


# --------------------------------------------------------------------------
# City registry. One entry per city.
#
# label_dates: the UNOSAT assessment dates to use, OLDEST FIRST. Listing more
#   than one date per city is what makes temporal label augmentation do
#   something: notebook 1 then preprocesses every date, and notebook 2 can
#   train on all of them with propagated labels. Use list_label_dates(city)
#   to see which assessment dates exist in your one_month folder.
# war_start: conflict start date, the pre war imagery window ends here.
# --------------------------------------------------------------------------

CITY_REGISTRY = {
    "Gaza": {
        "label_dates": [ "20240503", "20240706",
                        "20240906"], # "20231015", "20231107", "20231126", "20240106", "20240229", "20240401",
        "war_start": "2023-10-07",
    },
    # add the other cities once Gaza works end to end
    # "Aleppo":   {"label_dates": ["20160907"], "war_start": "2016-07-01"},
    # "Raqqa":    {"label_dates": ["20171021"], "war_start": "2017-06-06"},
    # "Mosul":    {"label_dates": ["20170804"], "war_start": "2016-10-16"},
    # "Mariupol": {"label_dates": ["20220512"], "war_start": "2022-02-24"},
    # "Rubizhne": {"label_dates": ["20220709"], "war_start": "2022-02-24"},
}


def add_city(name, label_dates, war_start):
    """Register another city, then re-run notebook 1 to preprocess it."""
    CITY_REGISTRY[name] = {"label_dates": list(label_dates), "war_start": war_start}


def list_label_dates(city, geometry=None):
    """All assessment dates for a city that exist in the one_month folder."""
    geometry = geometry or PREP["label_geometry"]
    pattern = os.path.join(DATA_DIR, f"{city}_*_1_{geometry}.csv")
    return sorted(os.path.basename(p).split("_")[1] for p in glob.glob(pattern))


# --------------------------------------------------------------------------
# Channel names. Selection everywhere happens by NAME, never by position,
# so a city exported with only SAR and a city with SAR plus optical both
# work without special cases.
# --------------------------------------------------------------------------

S1_CHANNELS = ["s1_pre_VV", "s1_pre_VH", "s1_post_VV", "s1_post_VH"]
S2_CHANNELS = ["s2_pre_B2", "s2_pre_B3", "s2_pre_B4", "s2_pre_B8",
               "s2_post_B2", "s2_post_B3", "s2_post_B4", "s2_post_B8"]

# PWTT columns that could serve as extra tabular features. damage_pts is
# banned on purpose: it counts the UNOSAT points that created the label,
# so using it as a feature would be leakage.
PWTT_BASELINE_COLUMN = "max_change"
PWTT_THRESHOLD = 3.3   # published decision threshold of the PWTT statistic


def wanted_channels(features):
    """Channel names a feature configuration asks for, in canonical order."""
    names = []
    for name in S1_CHANNELS + S2_CHANNELS:
        if name.startswith("s1") and not features["sentinel1"]:
            continue
        if name.startswith("s2") and not features["sentinel2"]:
            continue
        if "_pre_" in name and not features["pre_event"]:
            continue
        if "_post_" in name and not features["post_event"]:
            continue
        names.append(name)
    return names


def select_channels(city_data, features):
    """Slice a city stack down to the requested channels, resolved by name."""
    have = city_data["channel_names"]
    want = wanted_channels(features)
    missing = [n for n in want if n not in have]
    if missing:
        raise ValueError(
            f"{city_data['city']} was preprocessed with channels {have} but the "
            f"experiment asks for {missing}. Add the sensor to PREP['sensors'] "
            f"and re-run notebook 1.")
    idx = [have.index(n) for n in want]
    # A PatchSource, not an array: slicing a memory map with a channel list
    # would copy the whole file into RAM.
    return PatchSource.from_array(city_data["X"], idx)


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
    construction, which is what the temporal label matching in notebook 2
    relies on.
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
            f"{path} not found. Check that the one_month folder is on Drive "
            f"and that {city} has an assessment on {date}.")
    df = pd.read_csv(path, low_memory=False)
    geoms = [shape(json.loads(g)) for g in df[".geo"]]
    gdf = gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry=geoms, crs="EPSG:4326")
    # centroids are computed in a projected system, then converted back
    gdf["centroid"] = gdf.geometry.to_crs(3857).centroid.to_crs(4326)
    return gdf


# --------------------------------------------------------------------------
# File names for rasters and processed patches. The settings that shaped a
# file are baked into its name, so changing a setting forces a new file
# instead of silently reusing an old one.
# --------------------------------------------------------------------------

def raster_path(city, date, sensor):
    im = PREP["imagery"]
    os.makedirs(RASTER_DIR, exist_ok=True)
    tag = f"pre{im['pre_months']}m_{im['post_direction']}{im['post_months']}m"
    return os.path.join(RASTER_DIR, f"{city.lower()}_{date}_{sensor}_{tag}.tif")


def processed_paths(city, date):
    """(.npz metadata, .parquet geometries, .npy patches) for one city+date.

    The patches live in their own .npy rather than inside the .npz. A .npz is
    a zip archive, and a compressed zip member has no fixed position on disk,
    so it cannot be memory-mapped - reading one byte means decompressing the
    whole array. A bare .npy is a header plus raw contiguous values, so any
    patch can be located by arithmetic and read on its own.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    im = PREP["imagery"]
    tag = (f"p{PREP['patch_size']}_{PREP['label_geometry']}_n{PREP['n_sample']}"
           f"_{''.join(PREP['sensors'])}"
           f"_pre{im['pre_months']}m_{im['post_direction']}{im['post_months']}m")
    stem = os.path.join(PROCESSED_DIR, f"{city.lower()}_{date}_{tag}")
    return stem + ".npz", stem + ".parquet", stem + "_X.npy"


def save_processed(d):
    """Write one preprocessed city+date to disk (notebook 1 calls this).

    Patches go to their own uncompressed .npy as float16: float16 because
    backscatter in dB spans about -35 to +5, where it resolves ~0.01 dB, far
    finer than radar speckle; uncompressed and separate so the file can be
    memory-mapped when it is read back.
    """
    npz_path, parquet_path, x_path = processed_paths(d["city"], d["date"])
    np.save(x_path, d["X"].astype(np.float16))
    np.savez(npz_path, y=d["y"], lat=d["lat"], lon=d["lon"], xy=d["xy"],
             channel_names=np.array(d["channel_names"]))
    gdf = d["gdf"].drop(columns=["centroid"], errors="ignore")
    gdf.to_parquet(parquet_path)
    print(f"saved {os.path.basename(x_path)} "
          f"({os.path.getsize(x_path) / 1e9:.2f} GB)")


def load_processed(city, date=None):
    """Load one preprocessed city+date.

    X comes back as a float16 memory map: the pixels stay on disk and the
    operating system pages in only what is actually read, so several
    gigabytes of patches cost almost no memory. Wrap it with select_channels
    to get a PatchSource and let that do the reading, batch by batch. Never
    call np.asarray on the whole thing.
    """
    date = date or CITY_REGISTRY[city]["label_dates"][-1]
    npz_path, parquet_path, x_path = processed_paths(city, date)
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"{x_path} not found. Run notebook 1 with the current PREP "
            f"settings first. (Files saved by an older version of this "
            f"module kept the patches inside the .npz; re-run to convert.)")
    z = np.load(npz_path, allow_pickle=False)
    return {
        "city": city, "date": date,
        "X": np.load(x_path, mmap_mode="r"),
        "y": z["y"], "lat": z["lat"], "lon": z["lon"], "xy": z["xy"],
        "channel_names": [str(c) for c in z["channel_names"]],
        "gdf": gpd.read_parquet(parquet_path),
    }


def summarize_processed(city, dates=None):
    """Per-date row counts and damaged share, without touching any pixels.

    Reads only the small arrays in the .npz, so this is instant even when the
    patch files are gigabytes.
    """
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    rows = []
    for d in dates:
        npz_path, _, _ = processed_paths(city, d)
        with np.load(npz_path, allow_pickle=False) as z:
            y = z["y"]
        rows.append({"date": d, "buildings": len(y), "damaged": int(y.sum()),
                     "damaged %": round(y.mean() * 100, 2)})
    return pd.DataFrame(rows)


def shared_buildings(city, dates=None):
    """Building ids present at every date - what temporal augmentation uses."""
    dates = dates or CITY_REGISTRY[city]["label_dates"]
    sets = {}
    for d in dates:
        _, parquet_path, _ = processed_paths(city, d)
        # pandas, not geopandas: reading a single non-geometry column out of
        # a geoparquet file is exactly what geopandas refuses to do
        sets[d] = set(pd.read_parquet(parquet_path,
                                      columns=["system:index"])["system:index"])
    return set.intersection(*sets.values()), sets


def cut_patches(city, date, lonlat, patch=None):
    """Read patches straight from the rasters. Returns (X, keep_mask).

    About 200x slower than the stored patches, so this is for one-off
    inference or for re-cutting at a different patch_size without touching
    Earth Engine - never for the training loop.
    """
    patch = patch or PREP["patch_size"]
    half = patch // 2
    lonlat = np.asarray(lonlat, dtype=float)
    n = len(lonlat)
    blocks, keep = [], np.ones(n, dtype=bool)

    for sensor in PREP["sensors"]:
        with rasterio.open(raster_path(city, date, sensor)) as src:
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
# one latitude band, and (c) glue several dates together. Doing any of those
# with ordinary numpy indexing copies the whole array into memory, which
# undoes the memory mapping and is what runs Colab out of RAM.
#
# PatchSource records those operations instead of performing them, and reads
# pixels only when a batch is actually asked for.
# --------------------------------------------------------------------------

class PatchSource:
    """A lazy view over one or more patch arrays on disk."""

    def __init__(self, segments):
        # segments: list of (array, row indices, channel indices)
        self.segments = [(a, np.asarray(r, dtype=np.int64), list(c))
                         for a, r, c in segments]
        lengths = [len(r) for _, r, _ in self.segments]
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        arr0, _, ch0 = self.segments[0]
        self.shape = (int(self.offsets[-1]), len(ch0)) + arr0.shape[2:]
        self.dtype = arr0.dtype

    @classmethod
    def from_array(cls, array, channels=None):
        ch = list(range(array.shape[1])) if channels is None else list(channels)
        return cls([(array, np.arange(len(array)), ch)])

    def rows(self, mask_or_idx):
        """A new source keeping only these rows (boolean mask or indices)."""
        idx = np.asarray(mask_or_idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        segs = []
        for s, (arr, rws, ch) in enumerate(self.segments):
            lo, hi = self.offsets[s], self.offsets[s + 1]
            take = idx[(idx >= lo) & (idx < hi)] - lo
            if len(take):
                segs.append((arr, rws[take], ch))
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
            arr, rws, ch = self.segments[s]
            local = rws[gidx[here] - self.offsets[s]]
            # memory maps are much faster read in ascending order, so sort,
            # read, then put the rows back where the caller expects them
            order = np.argsort(local)
            block = np.asarray(arr[local[order]])[:, ch]
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

        Accumulating in float64 matters: summing millions of float16 values in
        float16 builds up serious rounding error.
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
# and hi are QUANTILES of building latitude. Bands carve one city into
# spatially disjoint regions: a random split would leak information through
# spatial autocorrelation (neighbouring buildings look alike and share
# their fate), a latitude band split cannot.
# --------------------------------------------------------------------------

def parse_split_entry(entry):
    if ":" in entry:
        city, band = entry.split(":")
        lo, hi = (float(v) for v in band.split("-"))
        return city, lo, hi
    return entry, 0.0, 1.0


def band_mask(lat, lo, hi):
    """Boolean mask selecting the latitude band between two quantiles."""
    qlo, qhi = np.quantile(lat, [lo, hi])
    return (lat >= qlo) & (lat <= qhi)


def split_assignment(lat, city, split_cfg):
    """Label every building of a city as train / val / test / unused."""
    role = np.array(["unused"] * len(lat), dtype=object)
    for part in ["train", "val", "test"]:
        for entry in split_cfg[part]:
            c, lo, hi = parse_split_entry(entry)
            if c == city:
                role[band_mask(lat, lo, hi)] = part
    return role


# --------------------------------------------------------------------------
# Augmentation 1: temporal label propagation.
#
# Uses the physics of destruction: a building marked damaged at one
# assessment date stays damaged at every later date, and strictly before
# its first damaged label it counts as intact. With several assessment
# dates per city this multiplies the usable training labels. With a single
# date it changes nothing.
# --------------------------------------------------------------------------

def expand_labels_over_time(frames):
    """Propagate damage labels across the assessment dates of one city.

    frames: list of (date_string, city_data) OLDEST FIRST, all for the same
    city, as returned by load_processed. Buildings are matched across dates
    by their PWTT id column "system:index".

    Returns {date: new_label_array} aligned with each frame's rows.
    """
    first_damaged = {}
    for date, d in frames:
        ids = d["gdf"]["system:index"].to_numpy()
        for b in ids[d["y"] == 1]:
            first_damaged.setdefault(b, date)   # keeps the EARLIEST date

    out = {}
    for date, d in frames:
        ids = d["gdf"]["system:index"].to_numpy()
        out[date] = np.array(
            [1 if (b in first_damaged and first_damaged[b] <= date) else 0
             for b in ids], dtype=int)
        changed = int((out[date] != d["y"]).sum())
        if changed:
            print(f"  {d['city']} {date}: {changed} labels changed by propagation")
    return out


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
# Notebook 2 therefore calls this once per region.
# --------------------------------------------------------------------------

def spatial_smooth(xy, scores, k=8, weight=0.3):
    """Blend scores with the mean of the k nearest neighbours' scores.

    xy: (n, 2) coordinates in metres (stored by notebook 1), so distances
    are physical distances.
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
# Models. A registry keyed by name: a new architecture is a class with
# forward(x) plus one decorator line, and it becomes selectable through
# CONFIG["model"] in notebook 2. Every model takes a (batch, channels,
# patch, patch) tensor and returns one logit per patch.
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
    """The from-scratch baseline (~25k parameters). All channels stacked."""

    def __init__(self, n_channels):
        super().__init__()
        self.features = nn.Sequential(conv_block(n_channels, 16),
                                      conv_block(16, 32), conv_block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

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

    def __init__(self, n_channels, width=32):
        super().__init__()
        assert n_channels % 2 == 0, "siamese needs matching pre and post channels"
        self.encoder = nn.Sequential(conv_block(n_channels // 2, width),
                                     conv_block(width, width * 2),
                                     conv_block(width * 2, width * 4),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
        feat = width * 4
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(3 * feat, 128),
                                  nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

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


def build_model(name, channel_names, device="cpu"):
    """Create a model from the registry, configured for the active channels."""
    model = MODEL_REGISTRY[name](n_channels=len(channel_names))
    if hasattr(model, "set_channel_split"):
        model.set_channel_split(channel_names)
    n = sum(p.numel() for p in model.parameters())
    print(f"model {name}: {n:,} parameters")
    return model.to(device)


@torch.no_grad()
def predict_probs(model, X, mu=None, sd=None, device="cpu", batch=256,
                  zero_channels=None):
    """Damage probabilities for a PatchSource or a plain array.

    Normalization happens here, one batch at a time, so a full-precision copy
    of the data never exists. Pass zero_channels to blank a channel group,
    which is how the modality ablation works.
    """
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = np.asarray(X[i:i + batch]).astype(np.float32)
        if mu is not None:
            xb = (xb - mu) / sd
        if zero_channels:
            xb[:, list(zero_channels)] = 0.0
        out.append(torch.sigmoid(model(torch.from_numpy(xb).to(device))).cpu().numpy())
    return np.concatenate(out)


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