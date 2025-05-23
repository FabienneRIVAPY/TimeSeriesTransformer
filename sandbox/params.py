# -------------------------------------------------------------------
# file path
# -------------------------------------------------------------------
file_path = "../data/DAprices_201810010000_202505140000_hourly.csv"

# -------------------------------------------------------------------
# TimeSeriesDataSet
# -------------------------------------------------------------------
max_prediction_length = 24
max_encoder_length = 7 * 24

time_idx = "hours_from_start"
target = "Day Ahead Preise D_LU"
group_ids = ["zone"]
min_prediction_length = 1
static_categoricals = ["zone"]
time_varying_known_reals = [
    "hours_from_start",
    "Prognostizierte Erzeugung PV und Wind",
]
time_varying_unknown_reals = [
    "Day Ahead Preise D_LU",
    "Erzeugung Erneuerbare",
    "Erzeugung Konventionelle",
    "Residuallast",
]
time_varying_unknown_categoricals = [
    "year",
    "day_of_week",
    "month",
    "hour",
    "is_holiday_or_weekend",
]
time_varying_known_categoricals = ["GWL"]
add_relative_time_idx = True
add_target_scales = True
add_encoder_length = True


# -------------------------------------------------------------------
# Model/Training/Trainer
# -------------------------------------------------------------------
batch_size = 64

max_epochs = 50
accelerator = "gpu"  # "cpu"
devices = 1
enable_model_summary = True
gradient_clip_val = 0.1

learning_rate = 0.001
hidden_size = 16  # 160
attention_head_size = 1  # 4
dropout = 0.1
hidden_continuous_size = 16  # 160
output_size = (
    7  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
)
log_interval = 10
reduce_on_plateau_patience = 4


# -------------------------------------------------------------------
# Early Stopping
# -------------------------------------------------------------------
monitor = "val_loss"
min_delta = 1e-4
patience = 20
mode = "min"
# -------------------------------------------------------------------
# LearningRateMonitor
# -------------------------------------------------------------------
logging_interval = "step"
