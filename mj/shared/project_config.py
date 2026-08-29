"""Sensor-independent label definitions and city metadata."""

POSITIVE_DAMAGE_CODES = (1, 2, 3)

# Assessment dates are ordered from oldest to newest. ``war_start`` marks the
# end of the pre-event imagery period. The role describes whether a city is
# used for model development or reserved as a whole-city holdout.
CITY_REGISTRY = {
    "Gaza": {
        "label_dates": ["20240503", "20240706", "20240906"],
        "war_start": "2023-10-07",
        "role": "develop",
    },
    "Raqqa": {
        "label_dates": ["20171021"],
        "war_start": "2017-06-06",
        "role": "holdout",
    },
    "Mosul": {
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
