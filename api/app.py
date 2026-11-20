import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.schema import CATEGORICAL_SPEC, NUMERIC_SPEC

app = FastAPI(
    title="Прогноз итоговой оценки G3",
    description="Оценка по анкете ученика. Промежуточные оценки G1 и G2 не используются.",
)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            return None
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class StudentInput(BaseModel):
    school: Literal["GP", "MS"] = Field(description=CATEGORICAL_SPEC["school"]["help"])
    sex: Literal["F", "M"] = Field(description=CATEGORICAL_SPEC["sex"]["help"])
    age: int = Field(ge=15, le=22, description=NUMERIC_SPEC["age"]["help"])
    address: Literal["U", "R"] = Field(description=CATEGORICAL_SPEC["address"]["help"])
    famsize: Literal["LE3", "GT3"] = Field(description=CATEGORICAL_SPEC["famsize"]["help"])
    Pstatus: Literal["T", "A"] = Field(description=CATEGORICAL_SPEC["Pstatus"]["help"])
    Medu: int = Field(ge=0, le=4, description=NUMERIC_SPEC["Medu"]["help"])
    Fedu: int = Field(ge=0, le=4, description=NUMERIC_SPEC["Fedu"]["help"])
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    reason: Literal["home", "reputation", "course", "other"]
    guardian: Literal["mother", "father", "other"]
    traveltime: int = Field(ge=1, le=4, description=NUMERIC_SPEC["traveltime"]["help"])
    studytime: int = Field(ge=1, le=4, description=NUMERIC_SPEC["studytime"]["help"])
    failures: int = Field(ge=0, le=3, description=NUMERIC_SPEC["failures"]["help"])
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]
    romantic: Literal["yes", "no"]
    famrel: int = Field(ge=1, le=5, description=NUMERIC_SPEC["famrel"]["help"])
    freetime: int = Field(ge=1, le=5, description=NUMERIC_SPEC["freetime"]["help"])
    goout: int = Field(ge=1, le=5, description=NUMERIC_SPEC["goout"]["help"])
    Dalc: int = Field(ge=1, le=5, description=NUMERIC_SPEC["Dalc"]["help"])
    Walc: int = Field(ge=1, le=5, description=NUMERIC_SPEC["Walc"]["help"])
    health: int = Field(ge=1, le=5, description=NUMERIC_SPEC["health"]["help"])
    absences: int = Field(ge=0, le=100, description=NUMERIC_SPEC["absences"]["help"])


@app.get("/health")
def health():
    return {"model_ready": MODEL_PATH.is_file()}


@app.post("/predict")
def predict(sample: StudentInput):
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Модель не найдена. Сначала запустите обучение.")
    row = sample.model_dump()
    frame = pd.DataFrame([{column: row[column] for column in bundle["columns"]}])
    grade = float(bundle["pipeline"].predict(frame)[0])
    return {
        "prediction": round(grade, 2),
        "unit": bundle.get("unit", "баллы по шкале 0–20"),
        "target": bundle.get("target", "G3"),
        "model": bundle.get("model_name"),
    }
