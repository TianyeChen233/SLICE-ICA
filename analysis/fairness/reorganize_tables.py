#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reorganize the long-format fairness CSV into clean per-section tables + summary matrix.

Reads:  local_data/fairness/fairness_llm_vendor_models_chainB_20260731.csv
Writes: local_data/fairness/tables/00_summary_matrix.csv ... 06_perclass_chi2.csv
        and rewrites the master CSV with sorted rows / ordered columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("local_data/fairness")
MASTER = ROOT / "fairness_llm_vendor_models_chainB_20260731.csv"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

MODEL_ORDER = ["hybrid_original", "deepseek-r1-distill-qwen-14b", "kimi-k2_6",
               "MiniMax-M2_1", "glm-5", "qwen3_5-27b"]
ATTR_ORDER = ["gender", "age_bin", "acquisition_bin"]
SEC_ORDER = ["overall", "group", "pairwise", "chi2", "perclass_group", "perclass_chi2"]

df = pd.read_csv(MASTER)
df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
df["attribute"] = pd.Categorical(df["attribute"], ATTR_ORDER + ["overall"], ordered=True)
df["section"] = pd.Categorical(df["section"], SEC_ORDER, ordered=True)

rate_cols = ["acc", "prec", "tpr", "fpr", "tnr", "f1", "ppp", "ppp_a", "ppp_b",
             "f1_a", "f1_b", "aod", "aod_signed", "di", "eop", "eod_tpr", "eod_fpr", "eod_mean"]
for c in rate_cols:
    df[c] = df[c].round(4)
df["chi2"] = df["chi2"].round(4)
df["p_value"] = df["p_value"].astype(float).round(6)

# ---- master: sorted + ordered columns ----
master_cols = ["section", "model", "attribute", "group", "group_a", "group_b", "test",
               "n_subjects", "threshold", "n", "n_a", "n_b", "tp", "tn", "fp", "fn",
               "acc", "prec", "tpr", "fpr", "tnr", "f1", "ppp", "ppp_a", "ppp_b", "f1_a", "f1_b",
               "aod", "aod_signed", "di", "eop", "eod_tpr", "eod_fpr", "eod_mean",
               "chi2", "df", "p_value", "bias_yes_no"]
master = df[master_cols].sort_values(["section", "model", "attribute", "group", "group_a", "test"])
master.to_csv(MASTER, index=False, encoding="utf-8")

# ---- 01 overall ----
ov = df[df["section"] == "overall"][
    ["model", "threshold", "n", "n_subjects", "acc", "prec", "tpr", "fpr", "tnr", "f1", "ppp",
     "tp", "tn", "fp", "fn"]].sort_values("model")
ov.to_csv(OUT / "01_overall_performance.csv", index=False, encoding="utf-8")

# ---- 02 group ----
gp = df[df["section"] == "group"][
    ["model", "attribute", "group", "n", "n_subjects", "acc", "prec", "tpr", "fpr", "tnr",
     "f1", "ppp", "tp", "tn", "fp", "fn"]].sort_values(["model", "attribute", "group"])
gp.to_csv(OUT / "02_group_performance.csv", index=False, encoding="utf-8")

# ---- 03 pairwise ----
pw = df[df["section"] == "pairwise"][
    ["model", "attribute", "group_a", "group_b", "n_a", "n_b", "ppp_a", "ppp_b", "f1_a", "f1_b",
     "aod", "aod_signed", "di", "eop", "eod_tpr", "eod_fpr", "eod_mean"]].sort_values(
    ["model", "attribute", "group_a", "group_b"])
pw.to_csv(OUT / "03_pairwise_fairness.csv", index=False, encoding="utf-8")

# ---- 04 chi2 ----
ch = df[df["section"] == "chi2"][
    ["model", "attribute", "test", "chi2", "df", "p_value", "bias_yes_no"]].sort_values(
    ["model", "attribute", "test"])
ch.to_csv(OUT / "04_chi2_tests.csv", index=False, encoding="utf-8")

# ---- 05 perclass group ----
pcg = df[df["section"] == "perclass_group"][
    ["model", "attribute", "group", "test", "n", "acc", "prec", "f1"]].rename(
    columns={"test": "class", "acc": "acc_within_class", "prec": "prec_class", "f1": "f1_class"})
pcg["class"] = pcg["class"].str.replace("_class", "", regex=False)
pcg = pcg.sort_values(["model", "attribute", "group", "class"])
pcg.to_csv(OUT / "05_perclass_performance.csv", index=False, encoding="utf-8")

# ---- 06 perclass chi2 ----
pcc = df[df["section"] == "perclass_chi2"][
    ["model", "attribute", "test", "chi2", "df", "p_value", "bias_yes_no"]].copy()
pcc["subset"] = pcc["test"].str.replace("correctness_x_group_within_", "", regex=False).str.replace("_true", "", regex=False)
pcc = pcc[["model", "attribute", "subset", "chi2", "df", "p_value", "bias_yes_no"]].sort_values(
    ["model", "attribute", "subset"])
pcc.to_csv(OUT / "06_perclass_chi2.csv", index=False, encoding="utf-8")

# ---- 00 summary matrix (paper-ready) ----
rows = []
for m in MODEL_ORDER:
    for a in ATTR_ORDER:
        p = pw[(pw["model"] == m) & (pw["attribute"] == a)]
        c = ch[(ch["model"] == m) & (ch["attribute"] == a)]
        q = pcc[(pcc["model"] == m) & (pcc["attribute"] == a)]
        row = {
            "model": m, "attribute": a,
            "max_aod": p["aod"].max(), "min_di": p["di"].min(),
            "max_eop": p["eop"].max(), "max_eod_mean": p["eod_mean"].max(),
            "chi2_corr_p": c.loc[c["test"] == "correctness_x_group", "p_value"].iloc[0],
            "chi2_corr_bias": c.loc[c["test"] == "correctness_x_group", "bias_yes_no"].iloc[0],
            "chi2_pred_p": c.loc[c["test"] == "prediction_x_group", "p_value"].iloc[0],
            "chi2_pred_bias": c.loc[c["test"] == "prediction_x_group", "bias_yes_no"].iloc[0],
            "noise_subset_p": q.loc[q["subset"] == "noise", "p_value"].iloc[0],
            "noise_subset_bias": q.loc[q["subset"] == "noise", "bias_yes_no"].iloc[0],
            "signal_subset_p": q.loc[q["subset"] == "signal", "p_value"].iloc[0],
            "signal_subset_bias": q.loc[q["subset"] == "signal", "bias_yes_no"].iloc[0],
        }
        rows.append(row)
sm = pd.DataFrame(rows)
for c in ["max_aod", "min_di", "max_eop", "max_eod_mean"]:
    sm[c] = sm[c].round(4)
for c in ["chi2_corr_p", "chi2_pred_p", "noise_subset_p", "signal_subset_p"]:
    sm[c] = sm[c].astype(float).round(6)
sm.to_csv(OUT / "00_summary_matrix.csv", index=False, encoding="utf-8")

print("written to", OUT)
for f in sorted(OUT.glob("*.csv")):
    print(" ", f.name, pd.read_csv(f).shape)
print("master:", master.shape)
