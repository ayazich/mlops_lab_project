# Прогноз итоговой оценки G3

Учебный ML-проект: по анкете ученика сервис предсказывает итоговую оценку `G3` по португальскому языку. Пользователь заполняет форму в Streamlit, интерфейс отправляет JSON в FastAPI, API вызывает сохранённую модель и возвращает оценку.

Исследование и сравнение моделей лежат в `notebooks/EDA_hw_Ayaz.ipynb` и `notebooks/hw_Ayaz_models.ipynb`. Ноутбуки не заменяют скрипты обучения.

## Задача и данные

Датасет — [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), файл `data/student-por.csv` (649 учеников, курс португальского языка, разделитель `;`).

Целевая переменная — `G3`, итоговая оценка по шкале 0–20.

В сервис не входят промежуточные оценки `G1` и `G2`. В EDA они исключены: это оценки того же курса до финала, и в сценарии «с G1/G2» модель почти целиком опирается на `G2` (на тесте MAE около 0.72). Для интерфейса нужен прогноз по тому, что известно заранее.

Модель использует остальные 30 признаков анкеты:

- числовые: `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`;
- категориальные: `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.

По EDA сильнее всего с низкой `G3` связан `failures`. Слабее, но значимо, связаны `studytime`, образование матери `Medu` и намерение получать высшее образование `higher`. `absences` статистически значим, но слаб. Категории кодируются внутри пайплайна (`OneHotEncoder`), числовые признаки масштабируются (`StandardScaler`).

## Качество

Скрипт сравнивает среднее, линейную регрессию, Ridge, Lasso и случайный лес. Выбор — по MAE на 5-fold кросс-валидации обучающей части (80%, `seed=42`). Лучшая модель — `RandomForest` (400 деревьев, `min_samples_leaf=3`).

На отложенных 20% (130 учеников):

| Метрика | Значение |
|---|---|
| MAE | 2.04 балла |
| RMSE | 2.77 балла |
| R² | 0.21 |

Среднее по обучающей выборке ошибается на тесте примерно на 2.39 балла MAE. Лес лучше этой планки, но по одной анкете оценка предсказывается слабо: это ориентир, а не точный балл экзамена. Точные цифры последнего запуска пишутся в `metrics.json` и `reports/model_report.txt`.

## Структура

```text
data/student-por.csv     исходные данные
src/prepare.py           разбиение на train/test
src/train.py             сравнение моделей и сохранение лучшей
src/schema.py            список признаков, подписи и допустимые значения
models/model.joblib      пайплайн для API (появляется после обучения)
reports/                 отчёт о сравнении моделей
api/app.py               FastAPI: /health и /predict
ui/app.py                форма Streamlit
tests/test_api.py        проверка API
params.yaml              доля теста, seed и гиперпараметры леса
dvc.yaml                 этапы prepare и train
```

## Установка

Нужны Git и Python 3.11 (с pip подойдёт и 3.12). Docker нужен только для запуска в контейнерах. `uv` — как в шаблоне курса; без него достаточно pip.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если установлен [uv](https://docs.astral.sh/uv/getting-started/installation/) и Python 3.11:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-groups
```

Дальше команды даны через `python` из активированного `.venv`. С uv их можно запускать как `uv run ...`.

## Обучение

```bash
python -m src.prepare
python -m src.train
```

Либо одной цепочкой, если установлен DVC (`pip install 'dvc>=3,<5'` или группа `dev` в uv). Если Git и DVC в каталоге ещё не инициализированы:

```bash
git init
dvc init
dvc repro
```

`prepare` читает `data/student-por.csv` и сохраняет `data/train.csv` и `data/test.csv` (эти два файла в Git не коммитятся). `train` сохраняет `models/model.joblib`, `metrics.json` и `reports/model_report.txt`, а параметры и метрики — в MLflow (`sqlite:///mlflow.db`).

Сравнить запуски:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Интерфейс: <http://localhost:5000>.

Проверка API:

```bash
pytest -q tests
```

## Запуск API и UI

В двух терминалах, после обучения:

```bash
uvicorn api.app:app --reload
streamlit run ui/app.py
```

- API и Swagger: <http://localhost:8000/docs>
- проверка модели: <http://localhost:8000/health>
- форма: <http://localhost:8501>

Через Docker Compose (модель уже должна лежать в `models/`, она копируется в образ API при сборке):

```bash
docker compose up --build
```

Внутри Compose интерфейс ходит в API по адресу `http://api:8000`. После нового обучения образ API нужно пересобрать.

## Пример запроса

`GET /health` при наличии файла модели:

```json
{"model_ready": true}
```

`POST /predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "school": "GP", "sex": "F", "age": 17, "address": "U",
    "famsize": "GT3", "Pstatus": "T", "Medu": 2, "Fedu": 2,
    "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "studytime": 2, "failures": 0,
    "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes",
    "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2,
    "health": 4, "absences": 4
  }'
```

Ответ:

```json
{
  "prediction": 13.65,
  "unit": "баллы по шкале 0–20",
  "target": "G3",
  "model": "RandomForest"
}
```

Число `prediction` зависит от обученной модели. Некорректные значения (например, возраст вне 15–22 или `failures` больше 3) возвращают HTTP 422. Если файла модели нет, `/predict` возвращает 503.
