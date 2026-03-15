from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

IN_FILE = Path("data/features/features.parquet")
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["weak", "prng", "csprng"]

FEATURE_COLUMNS = [
    "entropy",
    "chi2",
    "bit_balance",
    "byte_autocorr1",
    "bit_autocorr1",
    "spec_flatness",
    "spec_peak_ratio",
]

FEATURE_LABELS = {
    "entropy": "Shannon entropy (bytes)",
    "chi2": "Chi-square statistic (byte uniformity)",
    "bit_balance": "Bit balance",
    "byte_autocorr1": "Byte autocorrelation (lag 1)",
    "bit_autocorr1": "Bit autocorrelation (lag 1)",
    "spec_flatness": "Spectral flatness",
    "spec_peak_ratio": "Spectral peak / mean",
}

# Piirrä jakaumakuvia vain näistä, jos tarvitset niitä raporttiin.
FEATURES_FOR_DISTRIBUTIONS = [
    "entropy",
    "chi2",
    "spec_peak_ratio",
]


def configure_figure_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=1.0,
    )
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titlepad": 10,
        }
    )


def savefig(fig: plt.Figure, base: str) -> None:
    fig.savefig(OUT_DIR / f"{base}.png", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / f"{base}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def get_class_order(df: pd.DataFrame) -> list[str]:
    present = df["label3"].dropna().unique().tolist()
    preferred = [label for label in CLASS_ORDER if label in present]
    remaining = [label for label in sorted(present) if label not in preferred]
    return preferred + remaining


def load_clean_data() -> pd.DataFrame:
    df = pd.read_parquet(IN_FILE)
    required_cols = FEATURE_COLUMNS + ["label3", "stream_id"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["label3"]
    groups = df["stream_id"]

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Piirrä feature-jakaumat luokittain erillisissä paneeleissa.
    Tämä toimii paremmin kuin kaikkien luokkien pakottaminen samaan akseliin.
    """
    labels = get_class_order(df)

    for feature_col in FEATURES_FOR_DISTRIBUTIONS:
        display_name = FEATURE_LABELS[feature_col]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(labels),
            figsize=(10.5, 3.6),
            sharey=True,
            constrained_layout=True,
        )

        if len(labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            subset = df.loc[df["label3"] == label, feature_col]

            sns.histplot(
                x=subset,
                bins=20,
                stat="count",
                element="bars",
                fill=True,
                alpha=0.9,
                ax=ax,
            )

            ax.set_title(label)
            ax.set_xlabel(display_name)
            ax.set_ylabel("Count")
            sns.despine(ax=ax, trim=True)

        fig.suptitle(display_name, y=1.02)
        savefig(fig, f"dist_{feature_col}_by_label3_panels")


def train_models(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(df)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=3000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)

    lr_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": np.abs(lr.coef_).mean(axis=0),
        }
    ).sort_values("importance", ascending=True)

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    return y_test, lr_pred, rf_pred, lr_importance, rf_importance


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    base: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    savefig(fig, base)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    xlabel: str,
    base: str,
) -> None:
    plot_df = importance_df.copy().sort_values("importance", ascending=True)
    plot_df["feature"] = plot_df["feature"].map(FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="importance",
        y="feature",
        ax=ax,
    )

    max_val = float(plot_df["importance"].max())
    ax.set_xlim(0, max_val * 1.12)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")

    sns.despine(ax=ax, trim=True)
    savefig(fig, base)


def main() -> None:
    configure_figure_style()
    df = load_clean_data()

    y_test, lr_pred, rf_pred, lr_importance, rf_importance = train_models(df)
    labels = get_class_order(df)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=lr_pred,
        labels=labels,
        title="Confusion matrix: Logistic Regression",
        base="confusion_matrix_lr_3class",
    )

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=rf_pred,
        labels=labels,
        title="Confusion matrix: Random Forest",
        base="confusion_matrix_rf_3class",
    )

    plot_feature_importance(
        importance_df=lr_importance,
        title="Coefficient-based importance: Logistic Regression",
        xlabel="Mean absolute coefficient magnitude",
        base="feature_importance_lr",
    )

    plot_feature_importance(
        importance_df=rf_importance,
        title="Feature importance: Random Forest",
        xlabel="Feature importance",
        base="feature_importance_rf",
    )

    print(f"Saved figures to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

