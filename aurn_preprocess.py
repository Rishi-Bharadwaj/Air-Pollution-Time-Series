import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import rdata
import yaml
from tqdm import tqdm

AURN_POL_MAP = {
    "co":    ("CO (mg/m³)",    "CO"),
    "no2":   ("NO2 (µg/m³)",   "NO2"),
    "o3":    ("Ozone (µg/m³)", "Ozone"),
    "pm10":  ("PM10 (µg/m³)",  "PM10"),
    "pm2.5": ("PM2.5 (µg/m³)", "PM2.5"),
    "so2":   ("SO2 (µg/m³)",   "SO2"),
}


def rdata_to_dataframe(file_path):
    parsed = rdata.parser.parse_file(file_path)
    converted = rdata.conversion.convert(parsed)
    col_name = list(converted.keys())[0]
    df = pd.DataFrame(converted[col_name])
    df["date"] = (pd.to_datetime(df["date"], unit="s", utc=True)
                  .dt.tz_convert("Europe/London")
                  .dt.tz_localize(None))
    return df


def convert_file(args):
    fname, rdata_dir, csv_dir = args
    fpath = os.path.join(rdata_dir, fname)
    df = rdata_to_dataframe(fpath)
    out_path = os.path.join(csv_dir, fname.rsplit(".", 1)[0] + ".csv")
    df.to_csv(out_path, index=False)
    return out_path


def process_site(args):
    site, files, csv_dir, proc_dir, date_start, date_end, full_index = args
    dfs = [pd.read_csv(os.path.join(csv_dir, f)) for f in sorted(files)]
    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = (combined.dropna(subset=["date"])
                        .set_index("date")
                        .rename_axis("Timestamp")
                        .sort_index())
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.loc[date_start:date_end]
    if combined.empty:
        return site, 0
    combined.columns = combined.columns.str.lower().str.strip()

    saved = 0
    for aurn_col, (out_col, pol_name) in AURN_POL_MAP.items():
        if aurn_col not in combined.columns or combined[aurn_col].isna().all():
            continue
        out = (combined[[aurn_col]]
               .rename(columns={aurn_col: out_col})
               .reindex(full_index))
        out.to_csv(os.path.join(proc_dir, f"site_{site}_{pol_name}.csv"))
        saved += 1
    return site, saved


def main():
    parser = argparse.ArgumentParser(description="Preprocess AURN RData files.")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["aurn"]["preprocess"]

    do_zip      = cfg.get("zip", False)
    do_rdata    = cfg.get("rdata_conversion", True)
    zip_path    = cfg.get("zip_path")
    unzip_dir   = cfg["unzip_dir"]
    rdata_dir   = cfg["rdata_dir"]
    csv_dir     = cfg["csv_dir"]
    proc_dir    = cfg["output_dir"]
    date_start  = cfg["date_start"]
    date_end    = cfg["date_end"]
    max_workers = cfg.get("max_workers", 8)

    if do_zip:
        print(f"Unzipping {zip_path} -> {unzip_dir}")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(unzip_dir)
    else:
        print("Skipping unzip (zip: false in config)")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    if do_rdata:
        rdata_files = [f for f in os.listdir(rdata_dir)
                       if f.endswith((".RData", ".rdata", ".rds"))]
        print(f"Converting {len(rdata_files)} RData files to CSV...")
        convert_args = [(f, rdata_dir, csv_dir) for f in rdata_files]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(convert_file, a): a[0] for a in convert_args}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error converting {futures[future]}: {e}")
    else:
        print("Skipping RData conversion (rdata_conversion: false in config)")

    year_start = pd.to_datetime(date_start).year
    year_end   = pd.to_datetime(date_end).year

    site_files = {}
    for f in os.listdir(csv_dir):
        if not f.endswith(".csv"):
            continue
        parts = f[:-4].rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        site, year = parts[0], int(parts[1])
        if year_start <= year <= year_end:
            site_files.setdefault(site, []).append(f)

    full_index = pd.date_range(
        start=date_start, end=date_end, freq="h", name="Timestamp"
    )

    print(f"Processing {len(site_files)} sites with {max_workers} workers...")
    process_args = [
        (site, files, csv_dir, proc_dir, date_start, date_end, full_index)
        for site, files in site_files.items()
    ]
    sites_processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_site, a): a[0] for a in process_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sites"):
            site = futures[future]
            try:
                _, n = future.result()
                if n:
                    sites_processed += 1
            except Exception as e:
                print(f"Error processing {site}: {e}")

    print(f"Done. {sites_processed}/{len(site_files)} sites had data in range. Output in {proc_dir}/")


if __name__ == "__main__":
    main()