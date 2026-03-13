import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm


def generate_comparison_df(feature, folder, full_index):
    feature_key = feature.split(" ")[0]
    site_dict = {}
    files = [f for f in sorted(os.listdir(folder))
             if os.path.isfile(os.path.join(folder, f)) and feature_key in f]
    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp").reindex(full_index)
        site_dict[file] = df[feature]
    return pd.DataFrame(site_dict)


def plot_site_comparison_heatmap(df, feature_name, limit_dict, image_dir):
    nan_counts = df.isnull().sum().sort_values()
    df = df.reindex(columns=nan_counts.index)
    df_transposed = df.T

    fig, ax = plt.subplots(figsize=(20, len(df.columns) * 0.025))
    vmax = limit_dict.get(feature_name, df.max().max())
    sns.heatmap(
        df_transposed,
        cmap="YlOrRd",
        cbar_kws={"label": feature_name},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        mask=df_transposed.isna(),
        vmin=0,
        vmax=vmax,
    )

    years = pd.to_datetime(df.index).year.unique()
    year_starts = [df.index.get_loc(pd.Timestamp(f"{year}-01-01"))
                   for year in years if pd.Timestamp(f"{year}-01-01") in df.index]
    ax.set_xticks(year_starts)
    ax.set_xticklabels(years, rotation=0, fontsize=10)

    plt.title(f"{feature_name} Across Sites Over Time", fontsize=16, pad=20)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Site", fontsize=12)
    plt.tight_layout()

    safe = feature_name.split(" ")[0]
    plt.savefig(os.path.join(image_dir, f"{safe}.png"), dpi=500, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualise pollutant data across sites.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("dataset", help="Dataset key in the config (e.g. epa, cpcb)")
    args = parser.parse_args()

    with open(args.config) as f:
        all_cfg = yaml.safe_load(f)

    if args.dataset not in all_cfg:
        raise ValueError(f"Dataset '{args.dataset}' not found in {args.config}. Available: {list(all_cfg)}")
    cfg = all_cfg[args.dataset]

    folder     = cfg["folder"]
    full_index = pd.date_range(cfg["date_range"]["start"], cfg["date_range"]["end"], freq="h")
    dicts_dir  = cfg["output"]["dicts_dir"]
    image_dir  = cfg["output"]["image_dir"]
    features   = cfg["features"]
    limit_dict = cfg["pollutant_limits"]

    os.makedirs(dicts_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print("Building per-pollutant comparison DataFrames...")
    for feature in tqdm(features):
        safe = feature.split(" ")[0]
        feature_df = generate_comparison_df(feature, folder, full_index)
        feature_df.to_csv(os.path.join(dicts_dir, f"{safe}_df.csv"))

    print("Plotting heatmaps...")
    for feature in tqdm(features):
        safe = feature.split(" ")[0]
        feature_df = pd.read_csv(
            os.path.join(dicts_dir, f"{safe}_df.csv"), index_col=0, parse_dates=True
        )
        plot_site_comparison_heatmap(feature_df, feature, limit_dict, image_dir)

    print(f"Done. Images saved to {image_dir}/")


if __name__ == "__main__":
    main()
