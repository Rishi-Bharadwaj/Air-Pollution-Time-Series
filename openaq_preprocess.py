import argparse
import glob
import gzip
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone

import pandas as pd
import yaml
from tqdm import tqdm

# ── pollutant mapping (openaq name → standard column name, file suffix) ──
POL_MAP = {
    "pm25": ("PM2.5 (µg/m³)", "PM2.5"),
    "pm10": ("PM10 (µg/m³)",  "PM10"),
    "no2":  ("NO2 (µg/m³)",   "NO2"),
    "so2":  ("SO2 (µg/m³)",   "SO2"),
    "co":   ("CO (mg/m³)",    "CO"),
    "o3":   ("Ozone (µg/m³)", "Ozone"),
}

# ppm → target unit conversion factors
PPM_FACTOR = {
    "no2": 1.88 * 1000,   # ppm → µg/m³
    "so2": 2.62 * 1000,   # ppm → µg/m³
    "o3":  1.96 * 1000,   # ppm → µg/m³
    "co":  1.15,           # ppm → mg/m³
}
# µg/m³ → target unit (only CO needs conversion)
UGM3_FACTOR = {"co": 1 / 1000}  # µg/m³ → mg/m³


def convert_value(value, parameter, units):
    if units == "ppm" and parameter in PPM_FACTOR:
        return value * PPM_FACTOR[parameter]
    if units == "µg/m³" and parameter in UGM3_FACTOR:
        return value * UGM3_FACTOR[parameter]
    return value


def unzip_country(raw_dir, country, years, out_dir, max_workers=16):
    os.makedirs(out_dir, exist_ok=True)
    gz_files = []
    for year in years:
        pattern = os.path.join(raw_dir, country, "records", "csv.gz",
                               "locationid=*", f"year={year}", "month=*", "*.csv.gz")
        gz_files.extend(glob.glob(pattern))

    def _unzip_one(gz_path):
        out_path = os.path.join(out_dir, os.path.basename(gz_path).replace(".gz", ""))
        if os.path.exists(out_path):
            return 0
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return 1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(_unzip_one, gz_files), total=len(gz_files), desc="Unzipping"))
    print(f"Unzipped {sum(results)} new files ({len(gz_files)} total) to {out_dir}")


def process_location(loc_id, files, output_dir):
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        return loc_id, 0

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["parameter"].isin(POL_MAP)]
    if df.empty:
        return loc_id, 0

    # parse as UTC (handles mixed DST offsets), then convert to local
    df["Timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    offsets = df["datetime"].str.extract(r"([+-]\d{2}:\d{2})$")[0]
    most_common_offset = offsets.mode().iloc[0]
    h, m = int(most_common_offset[:3]), int(most_common_offset[0] + most_common_offset[4:])
    local_tz = timezone(timedelta(hours=h, minutes=m))
    df["Timestamp"] = df["Timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    saved = 0
    for param, group in df.groupby("parameter"):
        col_name, suffix = POL_MAP[param]

        g = group[["Timestamp", "value", "units"]].copy()
        g["value"] = g.apply(lambda r: convert_value(r["value"], param, r["units"]), axis=1)
        g = g.rename(columns={"value": col_name})[["Timestamp", col_name]]
        g = g.drop_duplicates(subset="Timestamp").set_index("Timestamp").sort_index()

        g.to_csv(os.path.join(output_dir, f"site_{loc_id}_{suffix}.csv"))
        saved += 1

    return loc_id, saved


def main():
    parser = argparse.ArgumentParser(description="Preprocess OpenAQ data for a country.")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("dataset", help="Dataset key in config (e.g. openaq_australia, openaq_brazil)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.dataset]["preprocess"]

    raw_dir    = cfg["raw_dir"]
    country    = cfg["country_name"]
    unzip_dir  = cfg["unzip_dir"]
    output_dir = cfg["output_dir"]
    start_year = cfg["start_year"]
    end_year   = cfg["end_year"]
    max_workers = cfg.get("max_workers", 8)
    years = range(start_year, end_year + 1)

    # Step 1: unzip gzips
    print(f"Step 1: Unzipping {country} ({start_year}-{end_year})...")
    unzip_country(raw_dir, country, years, unzip_dir)

    # Step 2: group CSVs by location
    all_csvs = glob.glob(os.path.join(unzip_dir, "location-*.csv"))
    loc_files = defaultdict(list)
    for f in all_csvs:
        m = re.match(r"location-(\d+)-", os.path.basename(f))
        if m:
            loc_files[m.group(1)].append(f)
    print(f"Found {len(all_csvs)} CSVs across {len(loc_files)} locations")

    # Step 3: process locations in parallel
    os.makedirs(output_dir, exist_ok=True)
    tasks = [(lid, files, output_dir) for lid, files in loc_files.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_location, *t): t[0] for t in tasks}
        results = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing locations"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Failed on location {futures[fut]}: {e}")

    total_files = sum(r[1] for r in results)
    print(f"Done. Saved {total_files} pollutant files across {len(results)} locations to {output_dir}")


if __name__ == "__main__":
    main()
