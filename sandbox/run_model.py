import sys

sys.path.append(
    "C:/Users/Anwender/Documents/GitHub/RiVaPy_development/TimeSeriesTransformer/"
)
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import GroupNormalizer
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

from src.preprocess_data import *
from src.postprocess_model import *
import params
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------
# load and preprocess data
# ------------------------------------------------------------------
print("load and preprocess data...")
data = load_and_clean_data(file_path=params.file_path)
time_df = process_timeseries(data)
training_cutoff = time_df[params.time_idx].max() - params.max_prediction_length
print("done")

# ------------------------------------------------------------------
# create dataloaders
# ------------------------------------------------------------------
print("create dataloaders...")
training = TimeSeriesDataSet(
    time_df[lambda x: x.hours_from_start <= training_cutoff],
    time_idx=params.time_idx,
    target=params.target,
    group_ids=params.group_ids,
    min_encoder_length=params.max_encoder_length // 2,
    max_encoder_length=params.max_encoder_length,
    min_prediction_length=params.min_prediction_length,
    max_prediction_length=params.max_prediction_length,
    static_categoricals=params.static_categoricals,
    time_varying_known_reals=params.time_varying_known_reals,
    time_varying_unknown_reals=params.time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(transformation=None),
    add_relative_time_idx=params.add_relative_time_idx,
    add_target_scales=params.add_target_scales,
    add_encoder_length=params.add_encoder_length,
)
validation = TimeSeriesDataSet.from_dataset(
    training, time_df, predict=True, stop_randomization=True
)

# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(
    train=True, batch_size=params.batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=params.batch_size * 10, num_workers=0
)
print("done")

# ------------------------------------------------------------------
# train model
# ------------------------------------------------------------------
print("train model...")
early_stop_callback = EarlyStopping(
    monitor=params.monitor,
    min_delta=params.min_delta,
    patience=params.patience,
    verbose=True,
    mode=params.mode,
)
lr_logger = LearningRateMonitor(logging_interval=params.logging_interval)
# logger = TensorBoardLogger("lightning_logs")

trainer = Trainer(
    max_epochs=params.max_epochs,
    accelerator=params.accelerator,
    devices=params.devices,
    enable_model_summary=params.enable_model_summary,
    gradient_clip_val=params.gradient_clip_val,
    callbacks=[lr_logger, early_stop_callback],
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=params.learning_rate,
    hidden_size=params.hidden_size,
    attention_head_size=params.attention_head_size,
    dropout=params.dropout,
    hidden_continuous_size=params.hidden_continuous_size,
    output_size=params.output_size,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=params.log_interval,
    reduce_on_plateau_patience=params.reduce_on_plateau_patience,
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
print("done")

# ------------------------------------------------------------------
# load and save the model
# ------------------------------------------------------------------
print("load and save the model...")
best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
save_path_to_file(best_model_path, "best_tft_path.txt")
print("done")
