import os
import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


def load_config(path: str, dataset: str, model: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)[dataset][f"{model}_analysis"]


def rmse_pw(a, p):
    return np.sqrt(np.mean((a - p) ** 2, axis=1))


def mae_pw(a, p):
    return np.mean(np.abs(a - p), axis=1)


def mase_pw(a, p, past, eps=1e-12):
    naive_mae = np.mean(np.abs(np.diff(past, axis=1)), axis=1)
    mae_vals = np.mean(np.abs(a - p), axis=1)
    with np.errstate(invalid="ignore"):
        out = mae_vals / naive_mae
    out[naive_mae < eps] = np.nan
    return out


def main(config_path: str, dataset: str, model: str):
    cfg = load_config(config_path, dataset, model)

    RESULTS_DIR       = cfg["results_dir"]
    OUT_DIR           = cfg["output_dir"]
    POLLUTANTS        = cfg["pollutants"]
    CONTEXT_LENGTH    = cfg["context_length"]
    PREDICTION_LENGTH = cfg["prediction_length"]
    ONLY_SUMMARY      = cfg.get("only_summary", False)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_dirs = sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )

    dirs_by_poll = {p: [] for p in POLLUTANTS}
    for d in all_dirs:
        poll = d.rsplit("_", 1)[1]
        site = d.rsplit("_", 1)[0]
        if poll in dirs_by_poll:
            dirs_by_poll[poll].append((site, d))

    # For each prediction step j, the 7 same-hour-of-day context indices are j, j+24, ..., j+144
    same_hour_idx = np.array(
        [[j + 24 * k for k in range(CONTEXT_LENGTH // 24)] for j in range(PREDICTION_LENGTH)]
    )  # (12, 7)

    per_window_dfs = {}

    for poll in POLLUTANTS:
        ttm_rows, mean_rows, median_rows = [], [], []

        for site_name, folder in tqdm(dirs_by_poll[poll], desc=f"Processing {poll}"):
            site_path = os.path.join(RESULTS_DIR, folder)

            dataset      = torch.load(os.path.join(site_path, "dataset.pt"), weights_only=False)
            preds_tensor = torch.load(os.path.join(site_path, "predictions.pt"), weights_only=False)
            if model == "ttm":
                with open(os.path.join(site_path, "scaler_params.pkl"), "rb") as f:
                    scaler = pkl.load(f)

            future_vals = dataset["future_values"].numpy()[:, :, 0]  # (N, 12)
            past_vals   = dataset["past_values"].numpy()[:, :, 0]    # (N, 168)
            preds_np    = preds_tensor.numpy()[:, :, 0]              # (N, 12)

            if model == "ttm":
                mean_  = scaler["mean_"][0]
                scale_ = scaler["scale_"][0]
                a = future_vals * scale_ + mean_
                p = preds_np    * scale_ + mean_
                c = past_vals   * scale_ + mean_
            else:
                a = future_vals
                p = preds_np
                c = past_vals

            N = a.shape[0]

            c_same_hour = c[:, same_hour_idx]          # (N, 12, 7)
            mean_bl   = np.mean(c_same_hour, axis=2)   # (N, 12)
            median_bl = np.median(c_same_hour, axis=2) # (N, 12)

            meta = {"site": [site_name] * N, "window_idx": np.arange(N)}

            ttm_rows.append(pd.DataFrame({
                **meta,
                "RMSE": rmse_pw(a, p),
                "MAE":  mae_pw(a, p),
                "MASE": mase_pw(a, p, c),
            }))
            mean_rows.append(pd.DataFrame({
                **meta,
                "RMSE": rmse_pw(a, mean_bl),
                "MAE":  mae_pw(a, mean_bl),
                "MASE": mase_pw(a, mean_bl, c),
            }))
            median_rows.append(pd.DataFrame({
                **meta,
                "RMSE": rmse_pw(a, median_bl),
                "MAE":  mae_pw(a, median_bl),
                "MASE": mase_pw(a, median_bl, c),
            }))

        per_window_dfs[(poll, "TTM")]            = pd.concat(ttm_rows,    ignore_index=True)
        per_window_dfs[(poll, "Mean Baseline")]  = pd.concat(mean_rows,   ignore_index=True)
        per_window_dfs[(poll, "Median Baseline")] = pd.concat(median_rows, ignore_index=True)

        print(f"  {poll}: {len(per_window_dfs[(poll, 'TTM')])} windows across {len(dirs_by_poll[poll])} sites")

    # ── Export per-window CSVs ───────────────────────────────────────────────
    if not ONLY_SUMMARY:
        for (poll, method), df in per_window_dfs.items():
            safe_poll   = poll.replace(".", "").replace(" ", "_")
            safe_method = method.replace(" ", "_").lower()
            df.to_csv(os.path.join(OUT_DIR, f"{safe_poll}_{safe_method}_per_window.csv"), index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_rows = []
    for (poll, method), df in per_window_dfs.items():
        summary_rows.append({
            "pollutant":    poll,
            "method":       method,
            "RMSE_mean":    df["RMSE"].mean(),
            "RMSE_median":  df["RMSE"].median(),
            "MAE_mean":     df["MAE"].mean(),
            "MAE_median":   df["MAE"].median(),
            "MASE_mean":    df["MASE"].mean(),
            "MASE_median":  df["MASE"].median(),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["pollutant", "method"])
        .reset_index(drop=True)
        .round(4)
    )
    summary_df.to_csv(os.path.join(OUT_DIR, "pollutant_summary.csv"), index=False)
    print(summary_df.to_string())
    print(f"\nExported {len(per_window_dfs)} per-window CSVs + summary to {OUT_DIR}")

    # ── Charts: one bar chart per pollutant × metric (18 total) ─────────────
    charts_dir = os.path.join(OUT_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    methods = ["Mean Baseline", "Median Baseline", "TTM"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    x       = np.arange(len(methods))

    for metric in ["RMSE", "MAE", "MASE"]:
        col = f"{metric}_mean"
        for poll in POLLUTANTS:
            values = [
                summary_df.loc[
                    (summary_df["pollutant"] == poll) & (summary_df["method"] == m), col
                ].values[0]
                for m in methods
            ]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(x, values, color=colors, width=0.5)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=10)
            ax.set_ylabel(f"Mean {metric}")
            ax.set_title(f"{poll} — {metric}")
            ax.set_ylim(0, max(values) * 1.15)
            fig.tight_layout()

            safe_poll = poll.replace(".", "").replace(" ", "_")
            fig.savefig(os.path.join(charts_dir, f"{safe_poll}_{metric.lower()}.png"), dpi=150)
            plt.close(fig)

    print(f"Saved 18 charts to {charts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("dataset")
    parser.add_argument("model", choices=["ttm", "chronos"])
    args = parser.parse_args()
    main(args.config, args.dataset, args.model)
