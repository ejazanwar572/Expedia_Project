from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "churn_flag"
SPLIT_DATE = "2019-05-09"

NUMERIC_FEATURES = [
    "coupon_flag",
    "pay_now_flag",
    "cancel_flag",
    "loyalty_tier",
    "total_visit_minutes",
    "total_visit_pages",
    "landing_pages_count",
    "search_pages_count",
    "property_pages_count",
    "bkg_confirmation_pages_count",
    "bounce_visits_count",
    "searched_destinations_count",
    "hotel_star_rating",
    "booking_month",
    "booking_dayofweek",
]

CATEGORICAL_FEATURES = [
    "customer_type",
    "platform",
    "marketing_channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["bk_date"] = pd.to_datetime(df["bk_date"], errors="coerce")
    df["cancel_date"] = pd.to_datetime(df["cancel_date"], errors="coerce")

    df["cancel_flag"] = df["cancel_flag"].fillna(0).astype(int)
    df["marketing_channel"] = df["marketing_channel"].fillna("Missing")
    df["total_visit_minutes"] = df["total_visit_minutes"].fillna(df["total_visit_minutes"].median())

    df["booking_month"] = df["bk_date"].dt.month.astype(int)
    df["booking_dayofweek"] = df["bk_date"].dt.dayofweek.astype(int)

    return df


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def evaluate(df: pd.DataFrame) -> dict[str, float]:
    train_mask = df["bk_date"] < pd.Timestamp(SPLIT_DATE)
    test_mask = ~train_mask

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    scores = pipeline.predict_proba(X_test)[:, 1]
    labels = (scores >= 0.5).astype(int)

    return {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Expedia churn workflow")
    parser.add_argument("csv_path", type=Path, help="Path to the case-study CSV file")
    args = parser.parse_args()

    df = load_and_prepare(args.csv_path)
    metrics = evaluate(df)

    print("Minimal churn workflow")
    print(f"Train rows: {metrics['train_rows']:,}")
    print(f"Test rows:  {metrics['test_rows']:,}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.3f}")
    print(f"Accuracy:   {metrics['accuracy']:.3%}")


if __name__ == "__main__":
    main()
