import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

import requests
import streamlit as st

from src.schema import CATEGORICAL_SPEC, NUMERIC_SPEC, example_payload

st.set_page_config(page_title="Прогноз оценки G3", layout="wide")
st.title("Прогноз итоговой оценки G3")
st.caption(
    "Курс португальского языка, шкала 0–20. "
    "Модель смотрит только на анкету ученика и не использует промежуточные оценки G1 и G2."
)

api_url = os.environ.get("API_URL", "http://localhost:8000")
defaults = example_payload()


def _field(name, defaults):
    if name in NUMERIC_SPEC:
        spec = NUMERIC_SPEC[name]
        return int(
            st.number_input(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                value=int(defaults[name]),
                step=1,
                help=spec["help"],
            )
        )
    spec = CATEGORICAL_SPEC[name]
    return st.selectbox(
        spec["label"],
        spec["choices"],
        index=spec["choices"].index(defaults[name]),
        help=spec["help"],
    )

with st.form("student"):
    st.subheader("Школа и ученик")
    school_cols = st.columns(4)
    values = {}
    for column, name in zip(school_cols, ["school", "sex", "age", "address"]):
        with column:
            values[name] = _field(name, defaults)

    st.subheader("Семья")
    family_cols = st.columns(4)
    for column, name in zip(family_cols, ["famsize", "Pstatus", "Medu", "Fedu"]):
        with column:
            values[name] = _field(name, defaults)
    job_cols = st.columns(3)
    for column, name in zip(job_cols, ["Mjob", "Fjob", "guardian"]):
        with column:
            values[name] = _field(name, defaults)

    st.subheader("Учёба")
    study_cols = st.columns(4)
    for column, name in zip(study_cols, ["reason", "traveltime", "studytime", "failures"]):
        with column:
            values[name] = _field(name, defaults)
    support_cols = st.columns(4)
    for column, name in zip(support_cols, ["schoolsup", "famsup", "paid", "higher"]):
        with column:
            values[name] = _field(name, defaults)

    st.subheader("Быт и досуг")
    life_cols = st.columns(4)
    for column, name in zip(life_cols, ["activities", "nursery", "internet", "romantic"]):
        with column:
            values[name] = _field(name, defaults)
    habit_cols = st.columns(4)
    for column, name in zip(habit_cols, ["famrel", "freetime", "goout", "health"]):
        with column:
            values[name] = _field(name, defaults)
    alcohol_cols = st.columns(3)
    for column, name in zip(alcohol_cols, ["Dalc", "Walc", "absences"]):
        with column:
            values[name] = _field(name, defaults)

    submitted = st.form_submit_button("Предсказать оценку", type="primary")

if submitted:
    try:
        response = requests.post(f"{api_url}/predict", json=values, timeout=10)
        if response.status_code == 422:
            st.error("Проверьте поля формы: сервер отклонил значения.")
            st.json(response.json())
        else:
            response.raise_for_status()
            body = response.json()
            grade = body["prediction"]
            st.metric("Прогноз G3", f"{grade:.2f}", help=body.get("unit", "баллы по шкале 0–20"))
            if grade >= 10:
                st.success(f"Оценка {grade:.2f} из 20 — не ниже проходных 10 баллов.")
            else:
                st.warning(f"Оценка {grade:.2f} из 20 — ниже проходных 10 баллов.")
            st.caption(
                "Типичная ошибка модели на тесте — около 2 баллов MAE. "
                "Это ориентир по анкете, а не точная оценка экзамена."
            )
    except requests.RequestException as error:
        st.error(f"API недоступен ({api_url}): {error}")
