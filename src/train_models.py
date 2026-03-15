from __future__ import annotations
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

IN_FILE = "data/features/features.parquet"

FEATURES = [
    "entropy", 
    "chi2", 
    "bit_balance",
    "byte_autocorr1",
    "bit_autocorr1",
    "spec_flatness", 
    "spec_peak_ratio",
]

def main() -> None:
    df = pd.read_parquet(IN_FILE)

    X = df[FEATURES]
    y = df["label3"]
    groups = df["stream_id"]

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    #Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    print("\n=== Logistic Regression (3-class) ===")
    print(classification_report(y_test, y_pred_lr, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr, labels=lr.classes_))

    #Random Forest
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n=== Random Forest (3-class) ===")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf, labels=rf.classes_))

    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nRF feature importances:\n", importances)

    # Save feature importances to CSV for results
    importances.to_csv("data/results/rf_feature_importance.csv")

if __name__ == "__main__":
    main()

