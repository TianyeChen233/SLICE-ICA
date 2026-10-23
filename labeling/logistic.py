"""Training-only logistic ICA scoring and signal-positive metrics."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, lr=0.05, epochs=700, l2=1e-4):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mu) / std
    w = np.zeros(Xn.shape[1], dtype=float)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(Xn @ w + b)
        g = p - y
        dw = (Xn.T @ g) / len(Xn) + l2 * w
        db = np.mean(g)
        w -= lr * dw
        b -= lr * db
    return {"w": w, "b": b, "mu": mu, "std": std}


def predict_prob(model: dict, X: np.ndarray) -> np.ndarray:
    Xn = (X - model["mu"]) / model["std"]
    return sigmoid(Xn @ model["w"] + model["b"])


def metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_pred = (p >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    return {
        "n": int(len(y_true)),
        "threshold": thr,
        "accuracy": acc,
        "precision_signal": pre,
        "recall_signal": rec,
        "f1_signal": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
