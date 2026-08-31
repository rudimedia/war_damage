# Detecting War Damage from Satellite Imagery

This repository contains a building-level war-damage classification pipeline for Gaza. It combines UNOSAT damage assessments with Microsoft Building Footprints and satellite imagery from Sentinel-1 SAR and PlanetScope. The project compares compact convolutional neural networks trained from scratch with a pretrained remote-sensing model, evaluates performance across locations and assessment dates, and produces maps and health-facility damage summaries.

The code is organized as a sequence of Jupyter notebooks supported by shared Python modules. The main experiments were developed for Google Colab, but paths can be overridden through environment variables for local execution.

## Data and modelling approach

The shared labelling pipeline converts UNOSAT point assessments into binary building labels. Microsoft building footprints of at most 50 m² are removed, and each selected UNOSAT damage point is buffered by 10 m before it is joined to intersecting buildings. UNOSAT classes for moderate damage, severe damage, and destruction are treated as damaged; all remaining buildings are treated as intact.

Two imagery pipelines use these labels:

- **Sentinel-1 SAR:** 32 × 32 pixel patches at 10 m resolution with pre- and post-event VV/VH channels. Models are trained and evaluated using the Gaza assessments from 3 May, 6 July, and 6 September 2024. The pipeline compares base and Siamese CNNs, optionally adds spatial and temporal XGBoost features, and evaluates transfer to Raqqa, Mosul, Chernihiv, and Rubizhne.
- **PlanetScope:** 32 × 32 pixel patches at 3 m resolution with pre- and post-event blue, green, red, and near-infrared channels. Experiments compare base and Siamese CNNs trained from scratch with a Siamese RSP ResNet-50 pretrained on MillionAID.

The Gaza experiments use latitude bands for training, stacking, validation, and final testing. This keeps nearby buildings in the same split and makes the final Gaza test set geographically distinct from the model-development bands.

The shared exploratory analysis combines the PlanetScope and Sentinel-1 inputs at footprint level. It examines optical and SAR changes across UNOSAT damage categories, including buildings newly classified as damaged between May and July; visualizes spectral signatures, NDVI change, SAR change, and matched RGB examples; measures local damage clustering; quantifies patch-overlap leakage under random and geographic splits; and compares the train, stacking, validation, and test bands.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── shared/
│   ├── project_config.py
│   ├── 0_build_labeled_footprints.ipynb
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 3_interactive_map.ipynb
│   ├── 4_hospital_overlay.ipynb
│   └── data/
│       ├── health/       # local health-facility input
│       └── unosat/       # included source assessment geodatabases
├── sentinel_sar/
│   ├── pipeline.py
│   ├── 1_preprocessing_eda.ipynb
│   ├── 2_training_evaluation.ipynb
│   ├── data/
│   └── experiments/
├── planet/
│   ├── pipeline.py
│   ├── 1_preprocessing_eda.ipynb
│   ├── 2_training_evaluation.ipynb
│   ├── 2_training_evaluation_pretrained_rsp_resnet50.ipynb
│   ├── data/
│   └── experiments/
└── images/
```

### `shared/`

- `project_config.py` defines the common positive-damage classes, assessment dates, conflict start dates, and development/holdout roles for each city.
- `0_build_labeled_footprints.ipynb` reads and deduplicates UNOSAT assessments, downloads and filters Microsoft Building Footprints, performs the spatial join, and validates the resulting per-date label tables.
- `1_exploratory_data_analysis.ipynb` is the extensive cross-sensor EDA. It joins the May and July UNOSAT categories to the extracted footprints, fuses PlanetScope and Sentinel-1 patches for the same buildings, analyzes optical and SAR changes by damage severity, maps local damage clustering, evaluates spatial leakage at both sensors' patch scales, and compares the geographic data splits.
- `3_interactive_map.ipynb` reloads a frozen experiment, scores all footprints for a selected city and date, and creates a Google Earth Engine overview, a publication-style PNG, and a Lonboard diagnostic map with building-level tooltips.
- `4_hospital_overlay.ipynb` links model predictions to health facilities, summarizes predicted facility damage over time, and compares facilities with surrounding buildings.
- `data/unosat/` contains the UNOSAT source assessments included with the repository. Other contents of `data/`, such as cached footprints, derived label tables, and health-facility data, remain local and are excluded from Git.

#### Included UNOSAT assessments

The unzipped File Geodatabases in `shared/data/unosat/` cover the main Gaza study area and the four secondary Sentinel-1 evaluation cities:

| Study area | Included source assessment(s) |
|---|---|
| Gaza Strip | 7 January, 1 April, 3 May, 6 July, and 6 September 2024 |
| Raqqa, Syria | `CE20130604SYR_Raqqa_Deir.gdb` |
| Mosul, Iraq | `Damage_assessment_Mosul_20170804.gdb` |
| Chernihiv, Ukraine | `CE20220223UKR_UNOSAT_Chernihiv_Damage.gdb` |
| Rubizhne, Ukraine | `UNOSAT_CE20220223UKR_Rubizhne_CDA_20220709.gdb` |

These are the original UNOSAT assessment sources used by the shared labelling notebook. Generated footprint caches and labelled outputs are intentionally not tracked.

#### Health-facility data

`shared/4_hospital_overlay.ipynb` expects the local GeoPackage `shared/data/health/gaza_health_facilities.gpkg`. It contains 101 geolocated facilities in Gaza: 28 hospitals and 73 primary-care facilities. Its fields include facility identifiers and names, facility tier and subtype, services, level, owner, governorate, and point geometry.

The notebook links each facility to nearby building footprints, applies the trained Sentinel-1 model for all three Gaza assessment dates, and reports both facility-level predictions and damage among surrounding buildings.

### `sentinel_sar/`

- `pipeline.py` contains Sentinel-1 paths and preprocessing settings, the city registry interface, raster and patch I/O, spatial splitting, CNN architectures, feature engineering, metrics, and experiment-loading utilities.
- `1_preprocessing_eda.ipynb` exports Sentinel-1 imagery from Google Earth Engine, cuts row-aligned building patches, checks data quality, and performs exploratory analysis.
- `2_training_evaluation.ipynb` trains the base and Siamese CNNs, runs optional Optuna hyperparameter searches, fits the spatial/temporal XGBoost second stage, selects models on validation ROC-AUC, and performs the sealed Gaza and city-level holdout evaluations.
- `data/` stores exported rasters and processed patch arrays.
- `experiments/<experiment_name>/` stores configurations, checkpoints, metrics, predictions, diagnostic figures, and Optuna studies.

### `planet/`

- `pipeline.py` defines the PlanetScope data contract, preprocessing and integrity checks, latitude-band splitting, datasets, CNN architectures, neighbour features, metrics, and experiment utilities.
- `1_preprocessing_eda.ipynb` converts the eight-band PlanetScope raster into aligned pre/post building patches and performs sensor-specific integrity checks and lightweight EDA. The more extensive joint optical/SAR analysis is in `shared/1_exploratory_data_analysis.ipynb`.
- `2_training_evaluation.ipynb` trains and compares the from-scratch base and Siamese CNN pipelines, including optional Optuna tuning and spatial XGBoost stacking.
- `2_training_evaluation_pretrained_rsp_resnet50.ipynb` trains and evaluates the MillionAID-pretrained RSP ResNet-50 comparison on the same spatial split.
- `data/` contains source rasters, processed datasets, and pretrained weights.
- `experiments/<experiment_name>/` follows the same output structure as the SAR experiments.

### Experiment output structure

A completed experiment generally contains:

```text
experiments/<experiment_name>/
├── config.json
├── models/          # fitted model and stacker checkpoints
├── metrics/         # selected parameters and evaluation tables
├── predictions/     # building-level scores and map inputs
├── figures/         # training and evaluation diagnostics
├── optuna.db
└── optuna_sampler.pkl
```

## Recommended execution order

1. Run `shared/0_build_labeled_footprints.ipynb` to create the common building-label tables.
2. Run both sensor preprocessing notebooks:
   - `sentinel_sar/1_preprocessing_eda.ipynb`, and
   - `planet/1_preprocessing_eda.ipynb`.
3. After preprocessing both sensors, run `shared/1_exploratory_data_analysis.ipynb` for the joint PlanetScope/Sentinel-1 EDA.
4. Run the corresponding training and evaluation notebook. For PlanetScope, the pretrained RSP ResNet-50 notebook is a separate comparison using the dataset created in step 2. Models can be extracted from corresponding experiment folders (with the exception of RSP ResNet-50 due to file size restrictions).
5. Run `shared/3_interactive_map.ipynb` to create overview and diagnostic maps from a completed experiment.
6. Optionally run `shared/4_hospital_overlay.ipynb` for the health-facility application.

## Environment

Install the shared Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Google Earth Engine authentication and access to the configured Earth Engine project are required for the Sentinel-1 exports and GEE-based map backgrounds. The UNOSAT geodatabases are kept under `shared/data/unosat/`. The health-facility GeoPackage, PlanetScope imagery, pretrained weights, processed arrays, and large model checkpoints are not distributed as ordinary Git source files and must be placed in the data locations expected by the notebooks.
