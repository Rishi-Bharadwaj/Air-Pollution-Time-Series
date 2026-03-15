import os
import tempfile
import pickle as pkl
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

warnings.filterwarnings("ignore")

_model = None
_cfg = None

POL_MAP = {
    "SO2":   "SO2 (µg/m³)",
    "PM2.5": "PM2.5 (µg/m³)",
    "Ozone": "Ozone (µg/m³)",
    "NO2":   "NO2 (µg/m³)",
    "CO":    "CO (mg/m³)",
    "PM10":  "PM10 (µg/m³)",
}


def load_config(path: str, dataset: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)[dataset]["ttm"]


def init_worker(cfg):
    global _model, _cfg
    _cfg = cfg
    model_cfg = TinyTimeMixerConfig(
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
    )
    _model = TinyTimeMixerForPrediction.from_pretrained(
        cfg["model_path"], config=model_cfg, ignore_mismatched_sizes=True
    )


def process_file(file):
    cfg = _cfg
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    timestamp_column = cfg["timestamp_column"]
    id_columns = cfg.get("id_columns", [])
    split_config = {"train": 1.0, "test": 0.0}

    site_name = os.path.splitext(file)[0]
    site_dir = os.path.join(output_dir, site_name)
    os.makedirs(site_dir, exist_ok=True)

    pol = site_name.split("_")[-1]
    target_columns = [POL_MAP[pol]]

    df = pd.read_csv(os.path.join(input_dir, file), parse_dates=[timestamp_column])

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, _, _ = get_datasets(tsp, df, split_config)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=cfg["batch_size"],
            seed=cfg["seed"],
            report_to="none",
        ),
    )

    predictions_np = trainer.predict(dset_train).predictions[0]

    past_list, future_list, timestamps = [], [], []
    for i in range(len(dset_train)):
        item = dset_train[i]
        past_list.append(item["past_values"])
        future_list.append(item["future_values"])
        timestamps.append(item["timestamp"])
    past_tensor = torch.stack(past_list)
    future_tensor = torch.stack(future_list)
    torch.save(
        {"past_values": past_tensor, "future_values": future_tensor, "timestamps": timestamps},
        os.path.join(site_dir, "dataset.pt"),
    )

    preds_tensor = torch.tensor(predictions_np, dtype=torch.float32)
    torch.save(preds_tensor, os.path.join(site_dir, "predictions.pt"))

    scaler_obj = tsp.target_scaler_dict["0"]
    scaler_params = {
        "mean_": scaler_obj.mean_.tolist(),
        "scale_": scaler_obj.scale_.tolist(),
        "target_columns": target_columns,
        "scaler_type": "standard",
    }
    with open(os.path.join(site_dir, "scaler_params.pkl"), "wb") as f:
        pkl.dump(scaler_params, f)

    return site_name, past_tensor.shape, preds_tensor.shape


def main(config_path: str, dataset: str):
    cfg = load_config(config_path, dataset)
    set_seed(cfg["seed"])

    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    max_workers = cfg.get("max_workers", 4)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(cfg,)) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sites"):
            try:
                site_name, past_shape, pred_shape = future.result()
                print(f"Saved {site_name}: past {past_shape}, preds {pred_shape}")
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTM zero-shot inference over all sites (parallel)")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("dataset", help="Dataset key in the config (e.g. epa, cpcb)")
    args = parser.parse_args()
    main(args.config, args.dataset)
