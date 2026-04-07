import argparse
import json
import os
import re
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

urllib3.disable_warnings()

BASE_URL    = "https://sinaica.inecc.gob.mx/pags/datGrafs.php"
STATION_URL = "https://sinaica.inecc.gob.mx/pags/datosRed.php"

# ppm → target unit; PM already in µg/m³
PARAM_INFO = {
    "PM2.5": {"col": "PM2.5 (µg/m³)",  "suffix": "PM2.5", "factor": 1.0},
    "PM10":  {"col": "PM10 (µg/m³)",   "suffix": "PM10",  "factor": 1.0},
    "O3":    {"col": "Ozone (µg/m³)",  "suffix": "Ozone", "factor": 1.96 * 1000},
    "NO2":   {"col": "NO2 (µg/m³)",    "suffix": "NO2",   "factor": 1.88 * 1000},
    "SO2":   {"col": "SO2 (µg/m³)",    "suffix": "SO2",   "factor": 2.62 * 1000},
    "CO":    {"col": "CO (mg/m³)",     "suffix": "CO",    "factor": 1.15},
}


def get_stations() -> pd.DataFrame:
    r = requests.get(STATION_URL, verify=False, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    select = soup.find("select", {"id": "selPickHeadEst"})
    return pd.DataFrame([
        {
            "station_id":   opt["value"],
            "station_name": opt.text.strip(),
            "raw_tokens":   opt.get("data-tokens", ""),
        }
        for opt in select.find_all("option")
        if opt.get("value", "").strip()
    ])


def get_year(session, station_id, parameter, year) -> pd.DataFrame:
    payload = {
        "estacionId": station_id,
        "param":      parameter,
        "fechaIni":   f"{year}-01-01",
        "rango":      f"{year}-12-31", # To the best of my understanding from the javascript webpage and the experiments I tried, rango is not actually range, it is end date.  
        "tipoDatos":  "",
    }
    r = session.post(
        BASE_URL, data=payload,
        headers={"Referer": "https://sinaica.inecc.gob.mx/"},
        verify=False, timeout=30,
    )
    r.raise_for_status()

    match = re.search(r"var dat\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        return pd.DataFrame()
    data = json.loads(match.group(1))
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.rename(columns={"id": "row_id", "fecha": "date", "hora": "hour",
                             "valor": "value", "bandO": "extra", "val": "valid"})
    df["datetime"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(pd.to_numeric(df["hour"], errors="coerce"), unit="h")
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["valid"] = pd.to_numeric(df["valid"], errors="coerce")
    return df[["datetime", "value", "valid"]]


def scrape_station(station_id, out_dir, years, target_index) -> str:
    session = requests.Session()
    saved   = 0
    skipped = 0

    for param, info in PARAM_INFO.items():
        out_path = os.path.join(out_dir, f"station_{station_id}_{info['suffix']}.csv")
        if os.path.exists(out_path):
            skipped += 1
            continue

        yearly = []
        for year in years:
            try:
                df = get_year(session, station_id, param, year)
                if not df.empty:
                    yearly.append(df)
            except Exception:
                pass
            sleep(0.3)

        if not yearly:
            continue

        combined = pd.concat(yearly, ignore_index=True)
        combined = combined.dropna(subset=["value"])
        combined = combined[combined["valid"] == 1]
        combined = combined.drop_duplicates(subset="datetime")
        combined = combined.set_index("datetime").sort_index()

        combined["value"] = combined["value"] * info["factor"]
        combined = combined[["value"]].rename(columns={"value": info["col"]})
        combined = combined.reindex(target_index)
        combined.index.name = "Timestamp"

        combined.to_csv(out_path)
        saved += 1

    if skipped == len(PARAM_INFO):
        return f"skip:{station_id}"
    if saved == 0:
        return f"empty:{station_id}"
    return f"save:{station_id}:{saved} new, {skipped} skipped"


def main():
    parser = argparse.ArgumentParser(description="Scrape and preprocess SINAICA hourly data.")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["sinaica"]["preprocess"]

    out_dir     = cfg["output_dir"]
    date_start  = cfg["date_start"]
    date_end    = cfg["date_end"]
    max_workers = cfg.get("max_workers", 16)

    start = pd.Timestamp(date_start)
    end   = pd.Timestamp(date_end)
    years = range(start.year, end.year + 1)
    target_index = pd.date_range(start=start, end=end, freq="h")

    os.makedirs(out_dir, exist_ok=True)

    print("Fetching station list...")
    df_stations = get_stations()
    df_stations.to_csv(os.path.join(out_dir, "sinaica_stations.csv"), index=False)
    print(f"Found {len(df_stations)} stations")

    saved_count, skipped_count, empty_count = 0, 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scrape_station, row["station_id"], out_dir, years, target_index): row["station_id"]
            for _, row in df_stations.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stations"):
            try:
                result = future.result()
                if result.startswith("save"):
                    saved_count += 1
                elif result.startswith("skip"):
                    skipped_count += 1
                else:
                    empty_count += 1
            except Exception as e:
                print(f"Error on station {futures[future]}: {e}")
                empty_count += 1

    print(f"\nDone. Saved: {saved_count} | Skipped (existing): {skipped_count} | No data: {empty_count}")
    print(f"Output in {out_dir}/")


if __name__ == "__main__":
    main()
