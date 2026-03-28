import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from statsmodels.tsa.seasonal import MSTL
from tqdm import tqdm

def mstl_impute(series: pd.Series, periods=(24, 168), robust=True) -> pd.Series:
    """
    Impute missing values in a time series using MSTL decomposition.
    
    Args:
        series  : pd.Series with DatetimeIndex and NaN gaps
        periods : tuple of seasonal periods (default: daily=24, weekly=168 for hourly data)
        robust  : use robust LOESS fitting (recommended for noisy sensor data)
    
    Returns:
        pd.Series with NaNs filled
    """
    
    # --- Step 1: Record which indices are missing ---
    # We want to only write back values at originally missing positions,
    # not disturb observed values.
    missing_mask = series.isna()
    
    if not missing_mask.any():
        return series  # nothing to impute
    
    # --- Step 2: Preliminary interpolation ---
    # MSTL cannot accept NaNs internally. We do a naive linear interpolation
    # purely so the decomposition can fit. This is just scaffolding — the
    # result of this fill is NOT used as the final imputed value.
    prelim_filled = series.interpolate(method='linear').ffill().bfill()
    # ffill/bfill handles NaNs at the edges (interpolate() can't fill leading/trailing NaNs)

    # --- Step 3: Fit MSTL ---
    # periods=(24, 168): extract both daily (24h) and weekly (7*24=168h) seasonalities
    # stl_kwargs are passed down to each internal STL fit
    mstl = MSTL(
        prelim_filled,
        periods=periods,
        iterate=2,
        stl_kwargs={"robust": robust}
    )
    result = mstl.fit()

    # --- Step 4: Extract and sum all seasonal components ---
    # result.seasonal is a DataFrame — one column per period
    # e.g., columns: ['seasonal_24', 'seasonal_168']
    # We sum across columns to get the combined seasonal signal
    total_seasonal = result.seasonal.sum(axis=1)

    # --- Step 5: Deseasonalize the ORIGINAL series (NaNs preserved) ---
    # By subtracting seasonal from the original (not prelim_filled),
    # gaps that were NaN stay NaN — we haven't contaminated them yet.
    deseasonalized = series - total_seasonal

    # --- Step 6: Interpolate only in the deseasonalized (simpler) space ---
    # The residual trend signal is much smoother than raw data,
    # making linear interpolation much more accurate here.
    deseasonalized_imputed = deseasonalized.interpolate(method='linear').ffill().bfill()

    # --- Step 7: Recompose — add seasonal back ---
    fully_imputed = deseasonalized_imputed + total_seasonal

    # --- Step 8: Write back ONLY to originally missing positions ---
    output = series.copy()
    output[missing_mask] = fully_imputed[missing_mask]
    output = output.clip(lower=0, upper=series.max())

    return output

def replace_negatives_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(df >= 0)


def get_sites_per_pollutant(dicts_dir, features, max_gap_hours, max_data_missing):
    sites_per_pollutant = {}

    for pol in features:
        safe_key = pol.split(" ")[0]

        df = pd.read_csv(
            os.path.join(dicts_dir, f"{safe_key}_df.csv"),
            index_col=0, parse_dates=True,
        )
        df = replace_negatives_with_nan(df)

        missing_pct = df.isnull().sum(axis=0) * 100 / len(df)
        valid_sites = []

        for col in df.columns:
            site_stem = col.rsplit(f"_{safe_key}", 1)[0]

            if missing_pct[col] > (max_data_missing):
                continue

            is_missing = df[col].isnull().values
            max_gap = cur = 0
            for m in is_missing:
                if m:
                    cur += 1
                    max_gap = max(max_gap, cur)
                else:
                    cur = 0

            if max_gap <= max_gap_hours:
                valid_sites.append(site_stem)

        sites_per_pollutant[pol] = valid_sites
        print(f"{pol}: {len(valid_sites)} valid sites")

    return sites_per_pollutant


def process_site_pollutant(site_stem, pol, input_dir, output_dir, date_start, date_end):
    formula = pol.split(" ")[0]
    path = os.path.join(input_dir, f"{site_stem}_{formula}.csv")
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    series = df.set_index("Timestamp")[pol]

    full_index = pd.date_range(date_start, date_end, freq="h")
    series = series.reindex(full_index)
    series.name = pol
    series = replace_negatives_with_nan(series)

    imputed = mstl_impute(series)

    imputed.index.name = "Timestamp"
    out = imputed.reset_index()
    out.to_csv(os.path.join(output_dir, f"{site_stem}_{formula}.csv"), index=False)
    return site_stem, pol


def main():
    parser = argparse.ArgumentParser(description="MSTL-impute site data.")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("dataset", help="Dataset key in config (e.g. cpcb, epa)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.dataset]["imputation"]

    dicts_dir     = cfg["dicts_dir"]
    input_dir     = cfg["input_dir"]
    output_dir    = cfg["output_dir"]
    max_gap_hours = cfg["max_gap_hours"]
    max_data_missing  = cfg["max_data_missing"]
    max_workers   = cfg.get("max_workers", 4)
    date_start    = cfg["date_range"]["start"]
    date_end      = cfg["date_range"]["end"]
    features      = cfg["features"]

    os.makedirs(output_dir, exist_ok=True)

    print("Filtering sites by quality criteria...")
    sites_per_pollutant = get_sites_per_pollutant(dicts_dir, features, max_gap_hours, max_data_missing)

    tasks = [(site, pol) for pol, sites in sites_per_pollutant.items() for site in sites]
    print(f"\nImputing {len(tasks)} (site, pollutant) pairs with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_site_pollutant, site, pol, input_dir, output_dir, date_start, date_end,
            ): (site, pol)
            for site, pol in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    print(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    main()
