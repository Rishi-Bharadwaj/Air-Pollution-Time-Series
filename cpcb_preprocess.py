import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

POLS = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]


def file_sort(string):
    parts = string.split("_")
    year = parts[0]
    site_no = int(parts[2])
    return (site_no, year)


def process_site(site, files, folder, output_dir):
    temp_list = []
    for file in sorted(files, key=file_sort):
        df = pd.read_csv(os.path.join(folder, file))
        temp_list.append(df)

    x_df = pd.concat(temp_list, ignore_index=True)
    x_df["Timestamp"] = pd.to_datetime(x_df["Timestamp"], errors="raise")
    x_df = x_df.dropna(subset=["Timestamp"])
    x_df = x_df.set_index("Timestamp").sort_index()
    x_df = x_df.resample("1h").median()

    site_stem = os.path.splitext(site)[0].replace("_15Min", "")
    for pol in POLS:
        if pol not in x_df.columns or x_df[pol].isna().all():
            continue
        formula = pol.split(" ")[0]
        pol_df = x_df[[pol]].reset_index()
        pol_df.to_csv(os.path.join(output_dir, f"{site_stem}_{formula}.csv"), index=False)
    return site


def main():
    parser = argparse.ArgumentParser(description="Preprocess CPCB site CSVs.")
    parser.add_argument("config", help="Path to config.yaml")
    # parser.add_argument("dataset", help="Dataset key in config (e.g. cpcb)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["cpcb"]["preprocess"]

    do_zip       = cfg.get("zip", True)
    zip_path     = cfg["zip_path"]
    raw_dir      = cfg["unzip_dir"]
    input_folder = cfg["input_folder"]
    output_dir   = cfg["output_dir"]
    max_workers  = cfg.get("max_workers", 8)

    if do_zip:
        print(f"Unzipping {zip_path} -> {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
    else:
        print("Skipping unzip (zip: false in config)")

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    files = sorted(files, key=file_sort)
    sites = sorted(set(s[5:] for s in files))

    file_dict = {site: [f for f in files if site in f] for site in sites}

    print(f"Processing {len(sites)} sites with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_site, site, file_list, input_folder, output_dir): site
            for site, file_list in file_dict.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sites"):
            site = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {site}: {e}")

    print(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    main()
