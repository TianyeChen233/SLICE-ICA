#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combined fairness metrics for 4 models on Chain B (proxy gold = final hybrid pseudo-labels).

Models: hybrid_original (qwen3.5-based, from earlier L3 run) + three vendor LLM hybrids
(deepseek-r1-distill-qwen-14b, kimi-k2_6, MiniMax-M2_1) inferred on PSEUDO_PKL_CACHE.

Attributes: gender, age_bin, acquisition_bin (from ic_table_with_acquisition.csv).
Metric definitions identical to run_fairness_chainB.py (Wang et al. IEEE Big Data 2024;
Wan et al. HDS 2025): per-group Acc/Prec/TPR/FPR/TNR/F1/PPP; pairwise AOD (abs+signed),
DI, EOP, EOD_TPR/FPR/mean; chi-square on correctness x group and prediction x group.

Output: ONE long-format CSV -> local_data/fairness/fairness_llm_vendor_models_chainB_20260731.csv
"""
from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

ROOT = Path("local_data/fairness")
OUT_CSV = ROOT / "fairness_llm_vendor_models_chainB_20260731.csv"
JOIN_DIR = ROOT / "results/metrics"

MODEL_FILES = {
    "hybrid_original": ROOT / "results/l3_infer/pseudo_flat__hybrid__per_sample.csv",
    "deepseek-r1-distill-qwen-14b": ROOT / "results/l3_infer_vendor/pseudo_flat__deepseek-r1-distill-qwen-14b__per_sample.csv",
    "kimi-k2_6": ROOT / "results/l3_infer_vendor/pseudo_flat__kimi-k2_6__per_sample.csv",
    "MiniMax-M2_1": ROOT / "results/l3_infer_vendor/pseudo_flat__MiniMax-M2_1__per_sample.csv",
    "glm-5": ROOT / "results/l3_infer_vendor/pseudo_flat__glm-5__per_sample.csv",
    "qwen3_5-27b": ROOT / "results/l3_infer_vendor/pseudo_flat__qwen3_5-27b__per_sample.csv",
}
ATTRS = ["gender", "age_bin", "acquisition_bin"]
MIN_N_PAIR = 50


def binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    nan = float("nan")
    prec = tp / (tp + fp) if (tp + fp) else nan
    rec = tp / (tp + fn) if (tp + fn) else nan
    f1 = 2 * prec * rec / (prec + rec) if (math.isfinite(prec) and math.isfinite(rec) and prec + rec > 0) else nan
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": (tp + tn) / n if n else nan,
        "prec": prec, "tpr": rec, "rec": rec,
        "fpr": fp / (fp + tn) if (fp + tn) else nan,
        "tnr": tn / (tn + fp) if (tn + fp) else nan,
        "f1": f1, "ppp": (tp + fp) / n if n else nan,
    }


def pair_metrics(ra: dict, rb: dict) -> dict:
    nan = float("nan")
    if not all(math.isfinite(ra[k]) and math.isfinite(rb[k]) for k in ("tpr", "fpr")):
        return {"aod": nan, "aod_signed": nan, "eop": nan, "eod_tpr": nan, "eod_fpr": nan, "eod_mean": nan}
    dtpr = abs(ra["tpr"] - rb["tpr"])
    dfpr = abs(ra["fpr"] - rb["fpr"])
    di = nan
    if ra["ppp"] > 0 and rb["ppp"] > 0:
        di = min(ra["ppp"] / rb["ppp"], rb["ppp"] / ra["ppp"])
    return {
        "aod": 0.5 * (dtpr + dfpr),
        "aod_signed": 0.5 * ((ra["fpr"] - rb["fpr"]) + (ra["tpr"] - rb["tpr"])),
        "di": di,
        "eop": dtpr,
        "eod_tpr": dtpr,
        "eod_fpr": dfpr,
        "eod_mean": 0.5 * (dtpr + dfpr),
    }


def chi2_test(table: np.ndarray):
    table = np.asarray(table, dtype=float)
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan"), 0
    row = table.sum(1, keepdims=True)
    col = table.sum(0, keepdims=True)
    n = table.sum()
    expected = row @ col / n
    if np.any(expected <= 0):
        return float("nan"), float("nan"), 0
    stat = float(((table - expected) ** 2 / expected).sum())
    df = int((table.shape[0] - 1) * (table.shape[1] - 1))
    return stat, float(chi2_dist.sf(stat, df)), df


def bias_flag(p):
    if not isinstance(p, float) or not math.isfinite(p):
        return "NA"
    return "Yes" if p < 0.05 else "No"


def main():
    ic = pd.read_csv(ROOT / "data/derived/ic_table_with_acquisition.csv")
    ic["subject"] = ic["subject"].astype(str)
    key = ic[["subject", "session", "ic", "gender", "age_bin", "acquisition", "acquisition_bin", "y_final_noise"]].copy()

    rows_out = []
    sanity = {}
    for model, path in MODEL_FILES.items():
        if not path.exists():
            raise SystemExit(f"missing {path}")
        pr = pd.read_csv(path)
        pr["subject"] = pr["subject"].astype(str)
        df = pr.merge(key, left_on=["subject", "session", "ic_index"],
                      right_on=["subject", "session", "ic"], how="inner")
        mismatch = int((df["gt_noise"] != df["y_final_noise"]).sum())
        sanity[model] = {"n_pred": len(pr), "n_joined": len(df), "label_mismatch": mismatch,
                         "n_subjects": int(df["subject"].nunique())}
        JOIN_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(JOIN_DIR / f"l3_vendor_joined_{model}.csv", index=False)

        thr = float(df["threshold"].iloc[0])
        yt_all = df["gt_noise"].to_numpy().astype(int)
        yp_all = df["pred_noise"].to_numpy().astype(int)
        ov = binary_rates(yt_all, yp_all)
        rows_out.append({"section": "overall", "model": model, "attribute": "overall",
                         "group": "", "group_a": "", "group_b": "", "test": "",
                         "n_subjects": sanity[model]["n_subjects"], "threshold": thr,
                         **{k: ov[k] for k in ("n", "tp", "tn", "fp", "fn", "acc", "prec", "tpr", "fpr", "tnr", "f1", "ppp")}})

        for attr in ATTRS:
            groups = sorted(df[attr].dropna().unique())
            perf = {}
            for g in groups:
                sub = df[df[attr] == g]
                r = binary_rates(sub["gt_noise"].to_numpy().astype(int),
                                 sub["pred_noise"].to_numpy().astype(int))
                r["n_subjects"] = int(sub["subject"].nunique())
                perf[g] = r
                rows_out.append({"section": "group", "model": model, "attribute": attr,
                                 "group": g, "group_a": "", "group_b": "", "test": "",
                                 "n_subjects": r["n_subjects"], "threshold": thr,
                                 **{k: r[k] for k in ("n", "tp", "tn", "fp", "fn", "acc", "prec", "tpr", "fpr", "tnr", "f1", "ppp")}})
            for a, b in combinations([g for g in groups if perf[g]["n"] >= MIN_N_PAIR], 2):
                pm = pair_metrics(perf[a], perf[b])
                rows_out.append({"section": "pairwise", "model": model, "attribute": attr,
                                 "group": "", "group_a": a, "group_b": b, "test": "",
                                 "n_subjects": "", "threshold": thr,
                                 "n_a": perf[a]["n"], "n_b": perf[b]["n"],
                                 "ppp_a": perf[a]["ppp"], "ppp_b": perf[b]["ppp"],
                                 "f1_a": perf[a]["f1"], "f1_b": perf[b]["f1"],
                                 **pm})
            # chi2: correctness x group
            t1 = np.zeros((len(groups), 2))
            for i, g in enumerate(groups):
                sub = df[df[attr] == g]
                corr = (sub["gt_noise"].to_numpy() == sub["pred_noise"].to_numpy())
                t1[i, 0] = corr.sum(); t1[i, 1] = (~corr).sum()
            s1, p1, d1 = chi2_test(t1)
            rows_out.append({"section": "chi2", "model": model, "attribute": attr,
                             "group": "|".join(groups), "group_a": "", "group_b": "",
                             "test": "correctness_x_group", "n_subjects": "", "threshold": thr,
                             "chi2": s1, "p_value": p1, "df": d1, "bias_yes_no": bias_flag(p1)})
            # chi2: prediction x group
            t2 = np.zeros((len(groups), 2))
            for i, g in enumerate(groups):
                sub = df[df[attr] == g]
                t2[i, 0] = int((sub["pred_noise"] == 0).sum())
                t2[i, 1] = int((sub["pred_noise"] == 1).sum())
            s2, p2, d2 = chi2_test(t2)
            rows_out.append({"section": "chi2", "model": model, "attribute": attr,
                             "group": "|".join(groups), "group_a": "", "group_b": "",
                             "test": "prediction_x_group", "n_subjects": "", "threshold": thr,
                             "chi2": s2, "p_value": p2, "df": d2, "bias_yes_no": bias_flag(p2)})

            # Wan-style per-class analysis (binary analog of per-disease breakdown):
            # within noise-true subset -> group TPR differences; within signal-true -> TNR/FPR
            for cls_val, cls_name in ((1, "noise"), (0, "signal")):
                for g in groups:
                    sub = df[df[attr] == g]
                    yt_g = sub["gt_noise"].to_numpy().astype(int)
                    yp_g = sub["pred_noise"].to_numpy().astype(int)
                    m = yt_g == cls_val
                    n_cls = int(m.sum())
                    if n_cls == 0:
                        continue
                    acc_cls = float((yt_g[m] == yp_g[m]).mean())
                    tp_c = int(((yt_g == cls_val) & (yp_g == cls_val)).sum())
                    fp_c = int(((yt_g != cls_val) & (yp_g == cls_val)).sum())
                    fn_c = int(((yt_g == cls_val) & (yp_g != cls_val)).sum())
                    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else float("nan")
                    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else float("nan")
                    f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)
                            if (math.isfinite(prec_c) and math.isfinite(rec_c) and prec_c + rec_c > 0)
                            else float("nan"))
                    rows_out.append({"section": "perclass_group", "model": model, "attribute": attr,
                                     "group": g, "group_a": "", "group_b": "",
                                     "test": cls_name + "_class", "n_subjects": "", "threshold": thr,
                                     "n": n_cls, "acc": acc_cls, "prec": prec_c, "tpr": rec_c, "f1": f1_c})
                t3 = np.zeros((len(groups), 2))
                for i, g in enumerate(groups):
                    sub = df[(df[attr] == g) & (df["gt_noise"] == cls_val)]
                    corr = (sub["gt_noise"].to_numpy() == sub["pred_noise"].to_numpy())
                    t3[i, 0] = corr.sum(); t3[i, 1] = (~corr).sum()
                s3, p3, d3 = chi2_test(t3)
                rows_out.append({"section": "perclass_chi2", "model": model, "attribute": attr,
                                 "group": "|".join(groups), "group_a": "", "group_b": "",
                                 "test": "correctness_x_group_within_" + cls_name + "_true",
                                 "n_subjects": "", "threshold": thr,
                                 "chi2": s3, "p_value": p3, "df": d3, "bias_yes_no": bias_flag(p3)})

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("saved:", OUT_CSV, "rows:", len(out))
    print(json.dumps(sanity, indent=2))

    # compact console summary
    pw = out[out["section"] == "pairwise"]
    for model in MODEL_FILES:
        sub = pw[pw["model"] == model]
        o = out[(out["section"] == "overall") & (out["model"] == model)].iloc[0]
        print(f"\n== {model}  thr={o['threshold']:.2f}  Acc={o['acc']:.4f} F1={o['f1']:.4f}")
        for attr in ATTRS:
            s = sub[sub["attribute"] == attr]
            ch = out[(out["section"] == "chi2") & (out["model"] == model) & (out["attribute"] == attr)]
            pc = ch[ch["test"] == "correctness_x_group"]["p_value"].iloc[0]
            pp = ch[ch["test"] == "prediction_x_group"]["p_value"].iloc[0]
            print(f"  {attr:16s} maxAOD={s['aod'].max():.4f} minDI={s['di'].min():.4f} "
                  f"maxEOP={s['eop'].max():.4f} maxEOD={s['eod_mean'].max():.4f} "
                  f"chi2_corr_p={pc:.3g} chi2_pred_p={pp:.3g}")


if __name__ == "__main__":
    main()
