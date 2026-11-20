import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_FEATURES, EXCLUDED, FEATURES, NUMERIC_FEATURES, TARGET

ALPHAS = np.logspace(-3, 3, 25)


def make_models(cfg):
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "LinReg": LinearRegression(),
        "Ridge": RidgeCV(alphas=ALPHAS),
        "Lasso": LassoCV(alphas=ALPHAS, max_iter=20000, random_state=cfg["random_state"]),
        "RandomForest": RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        ),
    }


def make_pipe(model):
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


def train():
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    model_cfg = cfg["model"]
    seed = cfg["data"]["seed"]
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    x_train, y_train = train_df[FEATURES], train_df[TARGET]
    x_test, y_test = test_df[FEATURES], test_df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    fitted = {}
    for name, model in make_models(model_cfg).items():
        pipe = make_pipe(model)
        scores = cross_validate(
            pipe,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        fitted[name] = (pipe, pred)
        rows.append(
            {
                "model": name,
                "cv_mae": float(-scores["test_mae"].mean()),
                "cv_rmse": float(-scores["test_rmse"].mean()),
                "cv_r2": float(scores["test_r2"].mean()),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "test_r2": float(r2_score(y_test, pred)),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("cv_mae")
    # Среднее — только ориентир. Среди обучаемых моделей берём лучшую по CV MAE.
    candidates = comparison[comparison["model"] != "DummyMean"]
    best_name = candidates.iloc[0]["model"]
    best_row = candidates.iloc[0]
    pipe, preds = fitted[best_name]

    metrics = {
        "mae": round(float(best_row["test_mae"]), 4),
        "rmse": round(float(best_row["test_rmse"]), 4),
        "r2": round(float(best_row["test_r2"]), 4),
        "cv_mae": round(float(best_row["cv_mae"]), 4),
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "columns": FEATURES,
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": EXCLUDED,
            "target": TARGET,
            "model_name": best_name,
            "unit": "баллы по шкале 0–20",
        },
        "models/model.joblib",
    )
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))

    Path("reports").mkdir(exist_ok=True)
    report = [
        "Прогноз итоговой оценки G3 (португальский язык, шкала 0–20).",
        "Признаки: анкета ученика. G1 и G2 исключены: это промежуточные оценки,",
        "которые почти полностью определяют G3 и неизвестны до конца курса.",
        f"Лучшая модель по CV MAE (без DummyMean): {best_name}",
        f"Тест: MAE={metrics['mae']}, RMSE={metrics['rmse']}, R2={metrics['r2']}",
        "",
        comparison.round(4).to_string(index=False),
        "",
        "Числовые признаки: " + ", ".join(NUMERIC_FEATURES),
        "Категориальные признаки: " + ", ".join(CATEGORICAL_FEATURES),
    ]
    report_path = Path("reports/model_report.txt")
    report_path.write_text("\n".join(report) + "\n")

    tracking = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment("student-g3")
    with mlflow.start_run():
        mlflow.log_params({**model_cfg, "selected_model": best_name, "seed": seed})
        mlflow.log_metrics(metrics)
        signature = infer_signature(x_test, preds)
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(str(report_path))
    print(comparison.round(4).to_string(index=False))
    print(f"Saved {best_name}: {metrics}")


if __name__ == "__main__":
    train()
