import argparse
import glob
import os
import re
import zipfile

import pandas as pd
import yaml
from tqdm import tqdm

pollutant_col = {
    "CO":    "CO (mg/m³)",
    "NO2":   "NO2 (µg/m³)",
    "O3":    "Ozone (µg/m³)",
    "PM10":  "PM10 (µg/m³)",
    "PM2.5": "PM2.5 (µg/m³)",
    "SO2":   "SO2 (µg/m³)",
}
pollutant_name = {k: v.split(" ")[0] for k, v in pollutant_col.items()}

KEEP_TYPES = set(pollutant_col.keys())


def file_date(f):
    m = re.search(r"(\d{8})", os.path.basename(f))
    return m.group(1) if m else ""


def main():
    parser = argparse.ArgumentParser(description="Preprocess CNEMC hourly CSVs.")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["cnemc"]["preprocess"]

    do_zip     = cfg.get("zip", False)
    zip_path   = cfg.get("zip_path")
    unzip_dir  = cfg["unzip_dir"]
    data_dir   = cfg["data_dir"]
    out_dir    = cfg["output_dir"]
    start_date = pd.to_datetime(cfg["start_date"]).strftime("%Y%m%d")
    end_date   = pd.to_datetime(cfg["end_date"]).strftime("%Y%m%d")

    if do_zip:
        print(f"Unzipping {zip_path} -> {unzip_dir}")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unzip_dir)
    else:
        print("Skipping unzip (zip: false in config)")

    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    day_files = [f for f in all_files if start_date <= file_date(f) <= end_date]
    print(f"Reading {len(day_files)} files into RAM...")

    big_df = pd.concat(
        [pd.read_csv(f) for f in tqdm(day_files)],
        ignore_index=True,
    )
    big_df = big_df[big_df["type"].isin(KEEP_TYPES)]
    big_df["Timestamp"] = (
        pd.to_datetime(big_df["date"].astype(str), format="%Y%m%d")
        + pd.to_timedelta(big_df["hour"], unit="h")
    )
    big_df = big_df.drop(columns=["date", "hour"])
    print(f"Loaded. Shape: {big_df.shape}")

    full_index = pd.date_range(
        start=pd.to_datetime(start_date, format="%Y%m%d"),
        end=pd.to_datetime(end_date, format="%Y%m%d") + pd.Timedelta(hours=23),
        freq="h",
        name="Timestamp",
    )

    meta_cols = {"type", "Timestamp"}
    sites = sorted(set(big_df.columns) - meta_cols)
    print(f"Found {len(sites)} unique sites")

    for site in tqdm(sites, desc="sites"):
        sub = big_df[["Timestamp", "type", site]].dropna(subset=[site])
        for pollutant, group in sub.groupby("type"):
            col = pollutant_col[pollutant]
            out = (
                group[["Timestamp", site]]
                .rename(columns={site: col})
                .set_index("Timestamp")
                .reindex(full_index)
            )
            out_path = os.path.join(out_dir, f"site_{site}_{pollutant_name[pollutant]}.csv")
            out.to_csv(out_path)

    print(f"Done. Output in {out_dir}/")


if __name__ == "__main__":
    main()