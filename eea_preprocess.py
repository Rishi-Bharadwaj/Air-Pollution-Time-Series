import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

EEA_POL_MAP = {
    1:    ("SO2 (µg/m³)",   "SO2"),
    5:    ("PM10 (µg/m³)",  "PM10"),
    7:    ("Ozone (µg/m³)", "Ozone"),
    8:    ("NO2 (µg/m³)",   "NO2"),
    10:   ("CO (mg/m³)",    "CO"),
    6001: ("PM2.5 (µg/m³)", "PM2.5"),
}
EXPECTED_UNITS = {
    1: "ug.m-3", 5: "ug.m-3", 7: "ug.m-3",
    8: "ug.m-3", 10: "mg.m-3", 6001: "ug.m-3",
}
COUNTRIES = {"FR", "DE"}


def extract_site_id(samplingpoint):
    part = samplingpoint.split("/")[1]
    if part.startswith("SPO.DE_"):
        return part[len("SPO.DE_"):].split("_")[0]
    elif part.startswith("SPO-"):
        return part[len("SPO-"):].rsplit("_", 1)[0]
    return part


def convert_parquet(fname, folder, suffix, base_out_dir, start_date, end_date):
    fpath = os.path.join(folder, fname)
    df = pd.read_parquet(fpath)
    if df.empty:
        return "skip"
    country = df["Samplingpoint"].iloc[0].split("/")[0]
    if country not in COUNTRIES:
        return "skip"
    df = df[(df["Start"] >= start_date) & (df["Start"] <= end_date)]
    if df.empty:
        return "skip"
    stem = fname.replace(".parquet", "")
    out_path = os.path.join(base_out_dir, country, "raw", f"{stem}_{suffix}.csv")
    df.to_csv(out_path, index=False)
    return "save"


def process_group(paths, proc_dir, full_index):
    dfs = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["Start"])
        df = df[df["Validity"] >= 1]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        return "skip"

    pol_id = int(combined["Pollutant"].iloc[0])
    if pol_id not in EEA_POL_MAP:
        return "skip"
    col_name, formula = EEA_POL_MAP[pol_id]

    unit = str(combined["Unit"].iloc[0])
    if unit != EXPECTED_UNITS[pol_id]:
        print(f"WARNING: unexpected unit '{unit}' for pollutant {pol_id}")

    site_id = extract_site_id(combined["Samplingpoint"].iloc[0])

    combined["Value"] = pd.to_numeric(combined["Value"], errors="coerce")
    out = (combined[["Start", "Value"]]
           .rename(columns={"Start": "Timestamp", "Value": col_name})
           .set_index("Timestamp")
           .sort_index())
    out = out[~out.index.duplicated(keep="first")]
    out = out.reindex(full_index)

    out.to_csv(os.path.join(proc_dir, f"site_{site_id}_{formula}.csv"))
    return "save"


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEA parquet files.")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["eea"]["preprocess"]

    do_convert      = cfg.get("convert_parquet", True)
    download_folders = cfg["download_folders"]
    base_out_dir    = cfg["base_out_dir"]
    start_date      = pd.Timestamp(cfg["start_date"])
    end_date        = pd.Timestamp(cfg["end_date"])
    max_workers     = cfg.get("max_workers", 16)

    for c in COUNTRIES:
        os.makedirs(os.path.join(base_out_dir, c, "raw"), exist_ok=True)

    # Step 1: parquet → CSV
    if do_convert:
        print("Step 1: converting parquet files to CSV...")
        saved, skipped = 0, 0
        for suffix, folder in download_folders.items():
            parquet_files = [f for f in os.listdir(folder) if f.endswith(".parquet")]
            print(f"  {suffix}: {len(parquet_files)} files")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        convert_parquet, f, folder, suffix,
                        base_out_dir, start_date, end_date
                    ): f for f in parquet_files
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=suffix):
                    try:
                        if future.result() == "save":
                            saved += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        print(f"Error on {futures[future]}: {e}")
                        skipped += 1
        print(f"  Saved {saved}, skipped {skipped}")
    else:
        print("Skipping parquet conversion (convert_parquet: false in config)")

    # Step 2: group by site+pollutant, filter validity, reindex
    print("Step 2: grouping and processing CSVs...")
    full_index = pd.date_range(start=start_date, end=end_date, freq="h", name="Timestamp")

    saved, skipped = 0, 0
    for country in sorted(COUNTRIES):
        input_dir = os.path.join(base_out_dir, country, "raw")
        proc_dir  = os.path.join(base_out_dir, country, "processed")
        os.makedirs(proc_dir, exist_ok=True)

        groups = defaultdict(list)
        for fname in os.listdir(input_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(input_dir, fname)
            peek = pd.read_csv(fpath, usecols=["Samplingpoint", "Pollutant"], nrows=1)
            if peek.empty:
                continue
            site_id = extract_site_id(peek["Samplingpoint"].iloc[0])
            pol_id  = int(peek["Pollutant"].iloc[0])
            groups[(site_id, pol_id)].append(fpath)

        print(f"  {country}: {len(os.listdir(input_dir))} files → {len(groups)} site+pollutant groups")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_group, paths, proc_dir, full_index): key
                for key, paths in groups.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=country):
                try:
                    if future.result() == "save":
                        saved += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Error on {futures[future]}: {e}")
                    skipped += 1

    print(f"Done. Saved {saved}, skipped {skipped}. Output in {base_out_dir}/{{FR,DE}}/processed/")


if __name__ == "__main__":
    main()