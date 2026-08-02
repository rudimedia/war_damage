"""
Export Sentinel-1 patches for EVERY city in the PWTT validation set.

Loops all *_footprints.csv, looks up each city's war_start + inference date from
config.py, builds the pre/post S1 stack the way PWTT does, and exports one or
more TFRecord shards per city to Google Drive.

Payload discipline (why this avoids the 10 MB request limit):
  * centroids and the AOI bbox are computed CLIENT-SIDE in plain Python, so the
    export request carries only lightweight Point nodes — never building polygons,
    and never a giant all-footprints FeatureCollection just to get the bounds.

Split is NOT applied here — everything lands flat in one Drive folder and the
training script routes files to train/val/test by city. Export once, re-split freely.
"""
import glob
import json
import pandas as pd
import ee
from pwtt import lee_filter
from config import parse_name, war_start_for

ee.Initialize(project="war-damage-504215")

CSV_DIR      = "sam/data/one_month"
DRIVE_FOLDER = "pwtt_cnn"
PATCH_RADIUS = 16          # -> 33x33 px ≈ 330 m at 10 m/px
SCALE        = 10          # Sentinel-1 native resolution
PRE_INTERVAL  = 12         # months of pre-war reference
POST_INTERVAL = 1          # months (the one_month dataset)
CHUNK         = 2500       # points per shard — keeps each request well under 10 MB

kernel = ee.Kernel.rectangle(PATCH_RADIUS, PATCH_RADIUS, "pixels")
SELECTORS = ["pre_VV", "pre_VH", "post_VV", "post_VH", "class", "idx"]


def centroid_lonlat(geo_json):
    """Mean of a geometry's outer ring — computed in Python, not on EE.
    Handles Point / Polygon / MultiPolygon / GeometryCollection, and returns
    None for a missing/blank/empty geometry. Sub-building precision is
    irrelevant for a 330 m patch."""
    if not isinstance(geo_json, str) or not geo_json.strip():
        return None                                    # NaN / empty cell
    g = json.loads(geo_json)
    t = g.get("type")
    if t == "GeometryCollection":
        return centroid_lonlat(json.dumps(g["geometries"][0]))
    coords = g.get("coordinates")
    if not coords:
        return None                                    # empty geometry
    if t == "Point":
        return coords[0], coords[1]
    ring = coords
    while isinstance(ring[0][0], (list, tuple)):        # unwrap Polygon / MultiPolygon
        ring = ring[0]
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    n = len(ring)
    return sum(c[0] for c in ring) / n, sum(c[1] for c in ring) / n


def city_points(df):
    """List of (lon, lat, class, idx) — all client-side. Rows whose geometry is
    missing or unparseable are dropped and counted, since they can't be placed."""
    out, skipped = [], 0
    for i, row in df.iterrows():
        c = centroid_lonlat(row[".geo"])
        if c is None:
            skipped += 1
            continue
        out.append((c[0], c[1], int(row["class"]), int(i)))
    if skipped:
        print(f"  (skipped {skipped} rows with missing/unparseable geometry)")
    return out


def build_stack(aoi, war_start, inference_start):
    """Pre/post mean VV+VH, pinned to S1's native 10 m grid."""
    war_start = ee.Date(war_start)
    inf_start = ee.Date(inference_start)

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filterBounds(aoi)
          .select(["VV", "VH"])
          .map(lee_filter)
          .map(lambda img: img.log()))

    s1_proj = s1.first().projection()   # native UTM @ 10 m for THIS city

    pre  = s1.filterDate(war_start.advance(-PRE_INTERVAL, "month"), war_start)
    post = s1.filterDate(inf_start, inf_start.advance(POST_INTERVAL, "month"))

    return (pre.mean().rename(["pre_VV", "pre_VH"])
            .addBands(post.mean().rename(["post_VV", "post_VH"]))
            .unmask(0)
            .toFloat()
            .setDefaultProjection(s1_proj))


def export_city(csv_path):
    city, inference_start = parse_name(csv_path)
    war_start = war_start_for(city)
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"{city:16s} SKIPPED — empty CSV file")
        return
    if df.empty:
        print(f"{city:16s} SKIPPED — 0 rows")
        return
    pts = city_points(df)
    if not pts:
        print(f"{city:16s} SKIPPED — no valid geometries")
        return

    # AOI (and thus UTM zone) from a client-side bbox — no giant FeatureCollection
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    aoi = ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])

    stack = build_stack(aoi, war_start, inference_start)
    patched = stack.neighborhoodToArray(kernel)

    n_chunks = (len(pts) + CHUNK - 1) // CHUNK
    print(f"{city:16s} n={len(pts):>7}  war_start={war_start}  "
          f"inference={inference_start}  chunks={n_chunks}")

    for k in range(n_chunks):
        chunk = pts[k * CHUNK:(k + 1) * CHUNK]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lon, lat]), {"class": c, "idx": idx})
            for lon, lat, c, idx in chunk
        ])
        samples = patched.sampleRegions(collection=fc, scale=SCALE,
                                        geometries=False, tileScale=4)
        ee.batch.Export.table.toDrive(
            collection=samples,
            description=f"{city}_{k:03d}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"{city}_{k:03d}",
            fileFormat="TFRecord",
            selectors=SELECTORS,
        ).start()


if __name__ == "__main__":
    csvs = sorted(glob.glob(f"{CSV_DIR}/*_footprints.csv"))
    print(f"Found {len(csvs)} cities.\n")
    for path in csvs:
        export_city(path)
    print("\nAll export tasks queued. Watch them in the GEE Tasks tab or via "
          "ee.batch.Task.list(). Big cities (Gaza, Kharkiv, Mosul) take longest.")