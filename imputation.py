import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from sklearn.impute import KNNImputer
from tqdm import tqdm


def safe_pol_name(pol):
    return (
        pol.replace("(", "")
           .replace(")", "")
           .replace("/", "_")
           .replace(" ", "_")
           .replace("µ", "u")
           .replace("³", "3")
    )


def get_sites(dicts_dir, features, max_gap_hours, min_data_pct):
    sites_by_pollutant_gap = {}
    sites_by_pollutant_missing = {}

    for pol in features:
        safe_key = pol.split(" ")[0]      # e.g. "PM2.5" — matches visualise.py naming
        safe_pol = safe_pol_name(pol)     # e.g. "PM2.5_ug_m3" — matches preprocess naming
        suffix = f"_{safe_pol}.csv"

        df = pd.read_csv(
            os.path.join(dicts_dir, f"{safe_key}_df.csv"),
            index_col=0, parse_dates=True,
        )

        missing_pct = df.isnull().sum(axis=0) * 100 / len(df)
        good_gap, good_missing = [], []

        for col in df.columns:
            site_stem = col[: -len(suffix)] if col.endswith(suffix) else col

            if missing_pct[col] <= (100 - min_data_pct):
                good_missing.append(site_stem)

            is_missing = df[col].isnull().values
            max_gap = cur = 0
            for m in is_missing:
                if m:
                    cur += 1
                    max_gap = max(max_gap, cur)
                else:
                    cur = 0

            if max_gap <= max_gap_hours:
                good_gap.append(site_stem)

        sites_by_pollutant_gap[pol] = set(good_gap)
        sites_by_pollutant_missing[pol] = set(good_missing)

        print(f"{pol}:")
        print(f"  - Max gap <= {max_gap_hours}h: {len(good_gap)} sites")
        print(f"  - Missing <= {100 - min_data_pct}%: {len(good_missing)} sites")

    sites_gap = set.intersection(*sites_by_pollutant_gap.values())
    sites_missing = set.intersection(*sites_by_pollutant_missing.values())
    sites = list(sites_gap & sites_missing)

    print(f"\nSites with max gap <= {max_gap_hours}h (all pollutants): {len(sites_gap)}")
    print(f"Sites with missing <= {100 - min_data_pct}% (all pollutants): {len(sites_missing)}")
    print(f"Sites meeting BOTH criteria: {len(sites)}")
    return sites


def process_site(site_stem, input_dir, output_dir, features, date_start, date_end, n_neighbors):
    frames = {}
    for pol in features:
        path = os.path.join(input_dir, f"{site_stem}_{safe_pol_name(pol)}.csv")
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        frames[pol] = df.set_index("Timestamp")[pol]

    full_index = pd.date_range(date_start, date_end, freq="h")
    df = pd.DataFrame(frames).reindex(full_index)

    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear
    df["dayofweek"] = df.index.dayofweek

    imputed = KNNImputer(n_neighbors=n_neighbors).fit_transform(df)
    df_imputed = pd.DataFrame(imputed, index=df.index, columns=df.columns)

    out = df_imputed[features].reset_index(names="Timestamp")
    out.to_csv(os.path.join(output_dir, f"{site_stem}.csv"), index=False)
    return site_stem


def main():
    parser = argparse.ArgumentParser(description="KNN-impute site data.")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("dataset", help="Dataset key in config (e.g. cpcb, epa)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.dataset]["imputation"]

    dicts_dir     = cfg["dicts_dir"]
    input_dir     = cfg["input_dir"]
    output_dir    = cfg["output_dir"]
    max_gap_hours = cfg["max_gap_hours"]
    min_data_pct  = cfg["min_data_pct"]
    n_neighbors   = cfg.get("n_neighbors", 5)
    max_workers   = cfg.get("max_workers", 4)
    date_start    = cfg["date_range"]["start"]
    date_end      = cfg["date_range"]["end"]
    features      = cfg["features"]

    os.makedirs(output_dir, exist_ok=True)

    print("Filtering sites by quality criteria...")
    sites = get_sites(dicts_dir, features, max_gap_hours, min_data_pct)

    print(f"\nImputing {len(sites)} sites with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_site, site, input_dir, output_dir,
                features, date_start, date_end, n_neighbors,
            ): site
            for site in sites
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    print(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    main()
