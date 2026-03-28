import argparse
import glob
import os
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

type_to_pol = {
    "hourly_42401": "Sulphur Dioxide",
    "hourly_88101": "PM2.5",
    "hourly_44201": "Ozone",
    "hourly_42602": "Nitrogen Dioxide",
    "hourly_42101": "Carbon Monoxide",
    "hourly_81102": "PM10",
}
original_units = {
    "Sulphur Dioxide":  "Sulfur dioxide Parts per billion",
    "PM2.5":            "PM2.5 - Local Conditions Micrograms/cubic meter (LC)",
    "Ozone":            "Ozone Parts per million",
    "Nitrogen Dioxide": "Nitrogen dioxide (NO2) Parts per billion",
    "Carbon Monoxide":  "Carbon monoxide Parts per million",
    "PM10":             "PM10 Total 0-10um STP Micrograms/cubic meter (25 C)",
}
conversion_factors = {
    "Sulphur Dioxide":  2.62,
    "PM2.5":            1.0,
    "Ozone":            1.96 * 1000,
    "Nitrogen Dioxide": 1.88,
    "Carbon Monoxide":  1.15,
    "PM10":             1.0,
}
who_names = {
    "Sulphur Dioxide":  "SO2 (µg/m³)",
    "PM2.5":            "PM2.5 (µg/m³)",
    "Ozone":            "Ozone (µg/m³)",
    "Nitrogen Dioxide": "NO2 (µg/m³)",
    "Carbon Monoxide":  "CO (mg/m³)",
    "PM10":             "PM10 (µg/m³)",
}


def _load_and_separate(path, pol, by_year_by_site_dir):
    df = pd.read_csv(path, low_memory=False)
    separate_and_filter(df, pol, by_year_by_site_dir)
    return path


def separate_and_filter(df, pol, by_year_by_site_dir):
    df.columns = df.columns.str.strip().str.replace("_", " ")
    df["Timestamp"] = pd.to_datetime(df["Date Local"] + " " + df["Time Local"], format="%Y-%m-%d %H:%M")
    s = df["Parameter Name"].iloc[0] + " " + df["Units of Measure"].iloc[0]
    year = pd.to_datetime(df["Date Local"].iloc[0]).year
    df[s] = df["Sample Measurement"]
    df["Key"] = df.apply(lambda x: (x["State Code"], x["County Code"], x["Site Num"]), axis=1)
    df_f = df[["Key", "Timestamp", s]]

    for site_id, group_df in df_f.groupby("Key"):
        safe_name = f"site_{site_id[0]}_{site_id[1]}_{site_id[2]}_{pol}_{year}"
        safe_name = safe_name.replace("/", "_").replace(" ", "_")
        group_df.drop(columns=["Key"]).to_csv(os.path.join(by_year_by_site_dir, f"{safe_name}.csv"), index=False)


def join_years(by_year_by_site_dir, by_site_all_years_dir, target_start, target_end):
    units_to_pol = {v: k for k, v in original_units.items()}
    target_index = pd.date_range(start=target_start, end=target_end, freq="h")

    groups = defaultdict(list)
    for fpath in glob.glob(os.path.join(by_year_by_site_dir, "*.csv")):
        fname = os.path.basename(fpath).replace(".csv", "")
        parts = fname.split("_")
        key = "_".join(parts[:-1])
        groups[key].append((int(parts[-1]), fpath))

    for key, file_list in tqdm(groups.items(), desc="Joining years"):
        file_list.sort(key=lambda x: x[0])
        dfs = [pd.read_csv(fp) for _, fp in file_list]
        combined = pd.concat(dfs, ignore_index=True)

        combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
        combined = combined.drop_duplicates(subset="Timestamp").set_index("Timestamp").sort_index()
        combined = combined.reindex(target_index)
        combined.index.name = "Timestamp"

        orig_col = next((c for c in combined.columns if c in units_to_pol), None)
        if orig_col is None:
            print(f"Could not detect pollutant column for key: {key}")
            continue
        pol_name = units_to_pol[orig_col]
        combined[orig_col] = combined[orig_col] * conversion_factors[pol_name]
        combined = combined.rename(columns={orig_col: who_names[pol_name]})

        pol_suffix = pol_name.replace(" ", "_")
        site_prefix = key[: -len(pol_suffix) - 1]
        formula = who_names[pol_name].split(" ")[0]
        combined.to_csv(os.path.join(by_site_all_years_dir, f"{site_prefix}_{formula}.csv"))


def main():
    parser = argparse.ArgumentParser(description="Preprocess EPA hourly CSVs.")
    parser.add_argument("config", help="Path to config.yaml")
    # parser.add_argument("dataset", help="Dataset key in config (e.g. epa)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["epa"]["preprocess"]

    do_zip       = cfg.get("zip", True)
    zip_path     = cfg.get("zip_path")
    unzip_dir         = cfg["unzip_dir"]
    by_year_by_site_dir   = cfg["by_year_by_site_dir"]
    by_site_all_years_dir = cfg["by_site_all_years_dir"]
    target_start = cfg["target_start"]
    target_end   = cfg["target_end"]
    start_year= pd.to_datetime(target_start).year
    end_year= pd.to_datetime(target_end).year
    years = range(start_year, end_year+1)

    if do_zip:
        print(f"Unzipping zips from {zip_path} -> {unzip_dir}")
        os.makedirs(unzip_dir, exist_ok=True)
        for fpath in glob.glob(os.path.join(zip_path, "*.zip")):
            if any(fpath.endswith(f"{year}.zip") for year in years):
                with zipfile.ZipFile(fpath, "r") as z:
                    for member in z.infolist():
                        member.filename = os.path.basename(member.filename)
                        if member.filename:
                            z.extract(member, unzip_dir)
    else:
        print("Skipping unzip (zip: false in config)")

    os.makedirs(by_year_by_site_dir, exist_ok=True)
    os.makedirs(by_site_all_years_dir, exist_ok=True)

    print("Step 1: splitting raw CSVs by site...")
    tasks = []
    for year in years:
        for type_code, pol in type_to_pol.items():
            path = os.path.join(unzip_dir, f"{type_code}_{year}.csv")
            if not os.path.exists(path):
                print(f"Skipping {path} (not found)")
                continue
            tasks.append((path, pol))

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_load_and_separate, path, pol, by_year_by_site_dir): path
                   for path, pol in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Splitting files"):
            try:
                future.result()
            except Exception as e:
                print(f"Failed on {futures[future]}: {e}")

    print("Step 2: joining years and converting units...")
    join_years(by_year_by_site_dir, by_site_all_years_dir, target_start, target_end)
    print(f"Done. Output in {by_site_all_years_dir}/")


if __name__ == "__main__":
    main()
