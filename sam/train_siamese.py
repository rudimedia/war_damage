"""
Siamese CNN for building-damage detection from Sentinel-1 patches.

Shared encoder embeds pre and post into the same feature space; the classifier
sees [f_pre, f_post, |f_pre - f_post|] so it learns the *change*, not the scene.

Files are routed to train / val / test / ood by CITY (from config.py), so whole
cities are held out — no building leaks across splits.
"""
import glob
import os
import numpy as np
import tensorflow as tf
from config import split_of

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATCH  = 33
BATCH  = 128
EPOCHS = 60
DATA_GLOB = "/Users/breitner/Library/CloudStorage/GoogleDrive-sam.kaenner@gmail.com/My Drive/pwtt_cnn/*.tfrecord.gz"     # flat folder of {City}_{chunk}.tfrecord.gz

# ---------------------------------------------------------------------------
# Route shards to splits by city name
# ---------------------------------------------------------------------------
def city_of(path):
    return os.path.basename(path).split("_")[0]

_files = glob.glob(DATA_GLOB)
assert _files, f"No TFRecords matched {DATA_GLOB} — did the exports finish?"
SPLIT_FILES = {s: [] for s in ("train", "val", "test", "ood")}
for f in _files:
    s = split_of(city_of(f))
    if s:
        SPLIT_FILES[s].append(f)
for s, fs in SPLIT_FILES.items():
    cities = sorted({city_of(f) for f in fs})
    print(f"{s:5s}: {len(fs):>3} shards  {cities}")
assert SPLIT_FILES["train"] and SPLIT_FILES["val"], "train/val are empty."

# ---------------------------------------------------------------------------
# Parse TFRecords -> ((pre, post), label)
# ---------------------------------------------------------------------------
BANDS = ["pre_VV", "pre_VH", "post_VV", "post_VH"]
_spec = {b: tf.io.FixedLenFeature([PATCH, PATCH], tf.float32) for b in BANDS}
_spec["class"] = tf.io.FixedLenFeature([], tf.int64)

def _clean(t):
    return tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))

def parse(ex):
    d = tf.io.parse_single_example(ex, _spec)
    pre  = _clean(tf.stack([d["pre_VV"],  d["pre_VH"]],  axis=-1))
    post = _clean(tf.stack([d["post_VV"], d["post_VH"]], axis=-1))
    return (pre, post), tf.cast(d["class"], tf.float32)

def make_ds(files, training=False):
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(8192)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(SPLIT_FILES["train"], training=True)
val_ds   = make_ds(SPLIT_FILES["val"])

# ---------------------------------------------------------------------------
# Per-channel standardization, adapted on TRAIN patches only (pre+post pooled)
# ---------------------------------------------------------------------------
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(make_ds(SPLIT_FILES["train"])
           .map(lambda xy, y: tf.concat([xy[0], xy[1]], axis=0))
           .take(300))

# ---------------------------------------------------------------------------
# Shared encoder + Siamese head
# ---------------------------------------------------------------------------
def build_encoder():
    inp = tf.keras.Input((PATCH, PATCH, 2))
    x = norm(inp)
    for f in (32, 32):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    for f in (64, 64):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, x, name="encoder")

def build_siamese():
    encoder = build_encoder()                       # built ONCE -> weights shared
    pre_in  = tf.keras.Input((PATCH, PATCH, 2), name="pre")
    post_in = tf.keras.Input((PATCH, PATCH, 2), name="post")
    f_pre, f_post = encoder(pre_in), encoder(post_in)
    diff = tf.keras.layers.Lambda(lambda t: tf.abs(t[0] - t[1]),
                                  name="abs_diff")([f_pre, f_post])
    x = tf.keras.layers.Concatenate()([f_pre, f_post, diff])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model([pre_in, post_in], out, name="siamese_damage")

model = build_siamese()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, gamma=2.0),
    metrics=[tf.keras.metrics.AUC(name="roc_auc"),
             tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                     patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("siamese_damage.keras",
                                       monitor="val_pr_auc", mode="max",
                                       save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                         factor=0.5, patience=5),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---------------------------------------------------------------------------
# Evaluate held-out splits. TEST is in-distribution; OOD is the Middle East
# cities (caveated baseline) — report them SEPARATELY, never averaged together.
# ---------------------------------------------------------------------------
def report(name, files):
    if not files:
        print(f"\n[{name}] no shards — skipped.")
        return
    ds = make_ds(files)
    y = np.concatenate([yb.numpy() for _, yb in ds], axis=0)
    p = model.predict(ds, verbose=0).ravel()
    roc = tf.keras.metrics.AUC()(y, p).numpy()
    pr  = tf.keras.metrics.AUC(curve="PR")(y, p).numpy()
    print(f"\n[{name}]  n={len(y)}  pos={int(y.sum())} ({y.mean():.2%})  "
          f"ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}")
    print(f"  {'thr':>5} {'prec':>6} {'recall':>7} {'f1':>6}")
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        pred = p >= thr
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        print(f"  {thr:5.2f} {prec:6.3f} {rec:7.3f} {f1:6.3f}")

report("VAL",  SPLIT_FILES["val"])
report("TEST", SPLIT_FILES["test"])
report("OOD (Middle East)", SPLIT_FILES["ood"])