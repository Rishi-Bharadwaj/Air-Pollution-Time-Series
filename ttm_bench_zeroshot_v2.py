import os
import tempfile
import pickle as pkl
import warnings
import argparse

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.get_model import get_model
warnings.filterwarnings("ignore")


def load_config(path: str, dataset: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)[dataset]["ttm"]


def load_model(cfg):
    zeroshot_model = get_model(
        cfg["model_path"],
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
        freq_prefix_tuning=False,
        freq=None,
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )
    return zeroshot_model


def zeroshot_eval(data, cfg, column_specifiers, split_config, model):
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config,use_frequency_token=model.config.resolution_prefix_tuning)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=cfg["batch_size"],
            seed=cfg["seed"],
            report_to="none",
        ),
    )

    predictions_np = trainer.predict(dset_train).predictions[0]
    return dset_train, predictions_np, tsp


def main(config_path: str, dataset: str):
    cfg = load_config(config_path, dataset)
    set_seed(cfg["seed"])

    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    timestamp_column = cfg["timestamp_column"]
    # target_columns = cfg["target_columns"]
    id_columns = cfg.get("id_columns", [])


    pol_map={
        "SO2":"SO2 (µg/m³)",
    "PM2.5":"PM2.5 (µg/m³)",
    "Ozone":"Ozone (µg/m³)",
    "NO2": "NO2 (µg/m³)",
    "CO":  "CO (mg/m³)",
    "PM10":"PM10 (µg/m³)"}
    split_config = {"train": 1.0, "test": 0.0}

    files = sorted(f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))

    model = load_model(cfg)

    for file in tqdm(files, desc="Processing sites"):
        site_name = os.path.splitext(file)[0]
        site_dir = os.path.join(output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)
        pol = site_name.split("_")[-1]
        target_columns = [pol_map[pol]]

        df = pd.read_csv(os.path.join(input_dir, file), parse_dates=[timestamp_column])

        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }

        dset_train, preds, tsp = zeroshot_eval(df, cfg, column_specifiers, split_config, model)
        torch.cuda.empty_cache()

        # Save dataset tensors
        past_list, future_list, timestamps = [], [], []
        for i in range(len(dset_train)):
            item = dset_train[i]
            past_list.append(item["past_values"])
            future_list.append(item["future_values"])
            timestamps.append(dset_train[i]["timestamp"])
        past_tensor = torch.stack(past_list)
        future_tensor = torch.stack(future_list)
        torch.save(
            {"past_values": past_tensor, "future_values": future_tensor, "timestamps": timestamps},
            os.path.join(site_dir, "dataset.pt"),
        )

        # Save predictions
        preds_tensor = torch.tensor(preds, dtype=torch.float32)
        torch.save(preds_tensor, os.path.join(site_dir, "predictions.pt"))

        # Save scaler params
        scaler_obj = tsp.target_scaler_dict["0"]
        scaler_params = {
            "mean_": scaler_obj.mean_.tolist(),
            "scale_": scaler_obj.scale_.tolist(),
            "target_columns": target_columns,
            "scaler_type": "standard",
        }
        with open(os.path.join(site_dir, "scaler_params.pkl"), "wb") as f:
            pkl.dump(scaler_params, f)

        print(f"Saved {site_name}: past {past_tensor.shape}, preds {preds_tensor.shape}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTM zero-shot inference over all sites")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("dataset", help="Dataset key in the config (e.g. epa, cpcb)")
    args = parser.parse_args()
    main(args.config, args.dataset)
