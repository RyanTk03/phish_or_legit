"""Lab 04 - evaluation: precision / recall / F1 of the PHISH class, confusion matrix, calibration.

Run from the lab folder:  python -m src.evaluate
"""
import json
import os

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, confusion_matrix, precision_recall_fscore_support)

from src.pipeline import load_data, split, make_pipeline, global_top_features


def ece(y, p, bins=10):
    """Expected calibration error: weighted mean |observed frequency - mean predicted score| per bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def main():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    yte = np.asarray(yte)
    print(f"train = {len(Xtr)} messages ({ytr.mean():.1%} phish) | test = {len(Xte)} messages "
          f"({yte.mean():.1%} phish)")
    print(f"Majority-class baseline accuracy (always 'legit'): {(1 - yte.mean()):.1%}  <- why accuracy alone is misleading")

    models = {
        "Naive Bayes": make_pipeline("nb"),
        "Logistic Regression": make_pipeline("lr"),
        "NB + calibration": make_pipeline("nb_cal"),
        "LR + char n-grams (extension)": make_pipeline("lr", analyzer="char_wb"),
    }
    results, probs = {}, {}
    print("\nmodel                           precision  recall    F1   accuracy  Brier   ECE")
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        acc = float((pred == yte).mean())
        results[name] = {"precision": pr, "recall": rc, "f1": f1, "accuracy": acc,
                         "brier": brier_score_loss(yte, p), "ece": ece(yte, p)}
        probs[name] = p
        r = results[name]
        print(f"{name:<32}{pr:>8.3f}{rc:>9.3f}{f1:>7.3f}{acc:>9.3f}{r['brier']:>8.3f}{r['ece']:>7.3f}")

    lr = models["Logistic Regression"]
    p_lr = probs["Logistic Regression"]
    cm = confusion_matrix(yte, (p_lr >= 0.5).astype(int))
    print("\nConfusion matrix (Logistic Regression)   rows = truth, columns = prediction (legit, phish)")
    print(cm)

    print("\nDecision threshold trade-off (Logistic Regression):")
    for th in (0.2, 0.35, 0.5, 0.7, 0.9):
        pred = (p_lr >= th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        print(f"  threshold {th:<4}: precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")

    pos, neg = global_top_features(lr)
    print("\nStrongest 'phish' features:", pos[:6])
    print("Strongest 'legit' features:", neg[:6])

    # --- failure cases
    texts = np.asarray(Xte)
    fn = np.where((yte == 1) & (p_lr < 0.5))[0]
    fp = np.where((yte == 0) & (p_lr >= 0.5))[0]
    lines = [f"False negatives (phish predicted legit): {len(fn)}"]
    lines += [f"  score={p_lr[i]:.2f} | {texts[i]}" for i in fn[:4]]
    lines += [f"False positives (legit predicted phish): {len(fp)}"]
    lines += [f"  score={p_lr[i]:.2f} | {texts[i]}" for i in fp[:4]]
    os.makedirs("results", exist_ok=True)
    open("results/failures.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n" + "\n".join(lines))

    json.dump({"models": results, "confusion_matrix_lr": cm.tolist()}, open("results/metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].imshow(cm, cmap="Greys")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["pred legit", "pred phish"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["true legit", "true phish"])
    axes[0].set_title("Confusion matrix (Logistic Regression)")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for (name, c) in (("Naive Bayes", "#8a8a8a"), ("Logistic Regression", "#333333"), ("NB + calibration", "#7a7a7a")):
        frac, mean = calibration_curve(yte, probs[name], n_bins=8, strategy="quantile")
        axes[1].plot(mean, frac, marker="o", color=c, label=name)
    axes[1].set_xlabel("mean predicted score"); axes[1].set_ylabel("observed share of phish")
    axes[1].set_title("Reliability diagram: is a score a probability?"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/phish_results.png", dpi=160)
    print("\nSaved results/metrics.json, results/failures.txt, results/phish_results.png")


if __name__ == "__main__":
    main()
