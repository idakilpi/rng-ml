from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

IN_FILE = Path("data/features/features.parquet")
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_MAIN = [
    ("entropy", "Shannon entropy (bytes)"),
    ("chi2", "Chi-square statistic (byte uniformity)"),
    ("bit_autocorr1", "Bit autocorrelation (lag 1)"),
    ("spec_flatness", "Spectral flatness"),
    ("spec_peak_ratio", "Spectral peak / mean"),
]

def set_academic_style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlepad": 10,
    })

def savefig(base: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{base}.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / f"{base}.pdf", bbox_inches="tight")
    plt.close()

def plot_feature_distributions(df: pd.DataFrame):
    for col, nice in FEATURES_MAIN:
        plt.figure(figsize=(8.5, 4.5))
        ax = sns.violinplot(
            data=df, x="label3", y=col,
            inner=None, cut=0
        )
        sns.boxplot(
            data=df, x="label3", y=col,
            width=0.25, showcaps=True, boxprops={"zorder": 2},
            showfliers=False, ax=ax
        )
        ax.set_title(f"{nice} by class")
        ax.set_xlabel("Class")
        ax.set_ylabel(nice)
        sns.despine()
        savefig(f"dist_{col}_by_label3")

def plot_faceted_overview(df: pd.DataFrame):
    keep_cols = [c for c, _ in FEATURES_MAIN]
    dlong = df[["label3"] + keep_cols].melt(id_vars="label3", var_name="feature", value_name="value")

    name_map = {c: nice for c, nice in FEATURES_MAIN}
    dlong["feature"] = dlong["feature"].map(name_map)

    g = sns.FacetGrid(dlong, col="feature", col_wrap=3, sharey=False, height=3.0)
    g.map_dataframe(sns.boxplot, x="label3", y="value", showfliers=False)
    g.set_axis_labels("Class", "")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=0)
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overview_boxplots.png", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "overview_boxplots.pdf", bbox_inches="tight", dpi=300)
    plt.close()

def plot_confusion_matrix_lr(df: pd.DataFrame):
   
    feature_cols = ["entropy", 
                    "chi2",
                    "bit_balance",
                    "byte_autocorr1", 
                    "bit_autocorr1",
                    "spec_flatness",
                    "spec_peak_ratio"]
    X = df[feature_cols]
    y = df["label3"]
    groups = df["stream_id"]

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_sc, y_train)
    pred = lr.predict(X_test_sc)

    labels = list(lr.classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)

    plt.figure(figsize=(6.2, 5.2))
    ax = sns.heatmap(cm, annot=True, fmt="d", cbar=True, square=True)
    ax.set_title("Confusion matrix (Logistic Regression, 3-class)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    sns.despine()
    savefig("confusion_matrix_lr_3class")

def main():
    set_academic_style()
    df = pd.read_parquet(IN_FILE)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    plot_feature_distributions(df)
    plot_faceted_overview(df)
    plot_confusion_matrix_lr(df)

    print(f"Saved figures to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

