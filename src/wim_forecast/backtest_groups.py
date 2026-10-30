"""Feature groups for ablations (kept free of lightgbm/torch imports)."""
from .build_dataset import LOGISTICS_CAL, SAME_DAY_EXOG

FEATURE_GROUPS = {
    "weather": SAME_DAY_EXOG + ["snow_lag1"],
    "calendar": ["dow", "month", "doy_sin", "doy_cos", "dom", "is_month_end", "is_holiday", "days_to_hol",
                 "days_from_hol", "is_bridge", "late_december", "covid_2020"],
    "ratio_lags": ["ratio_lag1", "ratio_lag2", "ratio_lag3", "ratio_lag7", "ratio_lag14",
                   "ratio_roll7", "ratio_roll28", "ratio_std28"],
    "truck_mix": ["combo_share_lag1", "tons_per_truck_lag1", "night_share_lag1"],
    "logistics_cal": LOGISTICS_CAL + ["holiday_on_weekday"],
    "mta": ["mta_ratio_lag1", "mta_ratio_roll7", "mta_ratio_lag7", "mta_carey_ratio_lag1", "mta_vz_ratio_lag1"],
    "morning": ["morning_ratio", "morning_combo_share", "morning_share_of_sched", "mta_morning_ratio"],
}


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")
