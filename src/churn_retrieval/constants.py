VEHICLE_FILE = "vehicle3.csv"
MEMBER_FILE = "member_info.csv"
REPAIR_FILE = "repare_maintain_info1.csv"

INTERNAL_REPAIR_PATTERN = r"内部|二手"
PASSIVE_REPAIR_PATTERN = r"事故|三包|质量担保|索赔|PDI|返工|免费|售前|代验车|召回|受控"

NUMERIC_COLUMNS = [
    "last_mile",
    "last_till_now_days",
    "first_to_purchase_day_diff",
    "first_to_purchase_mile_diff",
    "second_to_first_day_diff",
    "second_to_first_mile_diff",
    "day_diff_median",
    "mile_diff_median",
    "day_speed_median",
    "day_cv",
    "mile_cv",
    "day_speed_cv",
    "all_times",
    "car_age",
]

TEXT_COLUMNS = [
    "last_repair_type",
    "all_repair_types",
    "owner_type",
    "car_mode",
    "car_level",
    "member_level",
]
