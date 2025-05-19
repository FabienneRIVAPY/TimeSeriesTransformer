import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import GroupNormalizer
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from src.preprocess_data import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------
max_prediction_length = 24
max_encoder_length = 3 * 24
file_path = "data/DAprices_201810010000_202505140000_hourly.csv"

# ------------------------------------------------------------------
# load and preprocess data
# ------------------------------------------------------------------
print("load and preprocess data...")
data = load_and_clean_data(file_path=file_path)
time_df = process_timeseries(data)
training_cutoff = time_df["hours_from_start"].max() - max_prediction_length
print("done")

# ------------------------------------------------------------------
# create dataloaders
# ------------------------------------------------------------------
print("create dataloaders...")
training = TimeSeriesDataSet(
    time_df[lambda x: x.hours_from_start <= training_cutoff],
    time_idx="hours_from_start",
    target="DAprices",
    group_ids=["zone"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["zone"],
    time_varying_known_reals=[
        "hours_from_start",
        "day",
        "day_of_week",
        "month",
        "hour",
    ],
    time_varying_unknown_reals=["DAprices"],
    # target_normalizer=GroupNormalizer(
    #    groups=["zone"], transformation="softplus"
    # ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(
    training, time_df, predict=True, stop_randomization=True
)
# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)
print("done")

# ------------------------------------------------------------------
# train model
# ------------------------------------------------------------------
print("train model...")
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
)
lr_logger = LearningRateMonitor(logging_interval="step")
# logger = TensorBoardLogger("lightning_logs")

trainer = Trainer(
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=160,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
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
print("done")
