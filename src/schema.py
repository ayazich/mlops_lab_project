"""Признаки сервиса: анкета ученика без промежуточных оценок G1 и G2."""

TARGET = "G3"
EXCLUDED = ["G1", "G2"]

NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Подписи и допустимые значения для API и формы.
NUMERIC_SPEC = {
    "age": {
        "label": "Возраст",
        "help": "Полных лет",
        "min": 15,
        "max": 22,
        "default": 17,
    },
    "Medu": {
        "label": "Образование матери (Medu)",
        "help": "0 — нет, 1 — начальное, 2 — 5–9 классы, 3 — среднее, 4 — высшее",
        "min": 0,
        "max": 4,
        "default": 2,
    },
    "Fedu": {
        "label": "Образование отца (Fedu)",
        "help": "0 — нет, 1 — начальное, 2 — 5–9 классы, 3 — среднее, 4 — высшее",
        "min": 0,
        "max": 4,
        "default": 2,
    },
    "traveltime": {
        "label": "Дорога до школы (traveltime)",
        "help": "1 — до 15 мин, 2 — 15–30 мин, 3 — 30–60 мин, 4 — больше часа",
        "min": 1,
        "max": 4,
        "default": 1,
    },
    "studytime": {
        "label": "Время на учёбу в неделю (studytime)",
        "help": "1 — до 2 ч, 2 — 2–5 ч, 3 — 5–10 ч, 4 — больше 10 ч",
        "min": 1,
        "max": 4,
        "default": 2,
    },
    "failures": {
        "label": "Прошлые неудачи (failures)",
        "help": "Число прошлых неудовлетворительных периодов, 3 означает «3 и больше»",
        "min": 0,
        "max": 3,
        "default": 0,
    },
    "famrel": {
        "label": "Отношения в семье (famrel)",
        "help": "1 — очень плохие, 5 — отличные",
        "min": 1,
        "max": 5,
        "default": 4,
    },
    "freetime": {
        "label": "Свободное время (freetime)",
        "help": "1 — очень мало, 5 — очень много",
        "min": 1,
        "max": 5,
        "default": 3,
    },
    "goout": {
        "label": "Прогулки с друзьями (goout)",
        "help": "1 — очень редко, 5 — очень часто",
        "min": 1,
        "max": 5,
        "default": 3,
    },
    "Dalc": {
        "label": "Алкоголь по будням (Dalc)",
        "help": "1 — очень мало, 5 — очень много",
        "min": 1,
        "max": 5,
        "default": 1,
    },
    "Walc": {
        "label": "Алкоголь по выходным (Walc)",
        "help": "1 — очень мало, 5 — очень много",
        "min": 1,
        "max": 5,
        "default": 2,
    },
    "health": {
        "label": "Здоровье (health)",
        "help": "1 — очень плохое, 5 — очень хорошее",
        "min": 1,
        "max": 5,
        "default": 4,
    },
    "absences": {
        "label": "Число пропусков (absences)",
        "help": "В исходных данных от 0 до 32",
        "min": 0,
        "max": 100,
        "default": 4,
    },
}

CATEGORICAL_SPEC = {
    "school": {
        "label": "Школа",
        "help": "GP — Gabriel Pereira, MS — Mousinho da Silveira",
        "choices": ["GP", "MS"],
        "default": "GP",
    },
    "sex": {
        "label": "Пол",
        "help": "F — женский, M — мужской",
        "choices": ["F", "M"],
        "default": "F",
    },
    "address": {
        "label": "Тип адреса",
        "help": "U — город, R — село",
        "choices": ["U", "R"],
        "default": "U",
    },
    "famsize": {
        "label": "Размер семьи",
        "help": "LE3 — не больше 3 человек, GT3 — больше 3",
        "choices": ["GT3", "LE3"],
        "default": "GT3",
    },
    "Pstatus": {
        "label": "Родители живут вместе",
        "help": "T — вместе, A — раздельно",
        "choices": ["T", "A"],
        "default": "T",
    },
    "Mjob": {
        "label": "Работа матери",
        "help": "teacher, health, services, at_home, other",
        "choices": ["teacher", "health", "services", "at_home", "other"],
        "default": "other",
    },
    "Fjob": {
        "label": "Работа отца",
        "help": "teacher, health, services, at_home, other",
        "choices": ["teacher", "health", "services", "at_home", "other"],
        "default": "other",
    },
    "reason": {
        "label": "Причина выбора школы",
        "help": "home — близко к дому, reputation — репутация, course — программа, other — другое",
        "choices": ["home", "reputation", "course", "other"],
        "default": "course",
    },
    "guardian": {
        "label": "Опекун",
        "help": "mother, father или other",
        "choices": ["mother", "father", "other"],
        "default": "mother",
    },
    "schoolsup": {
        "label": "Дополнительная школьная поддержка",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "no",
    },
    "famsup": {
        "label": "Помощь семьи с учёбой",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "yes",
    },
    "paid": {
        "label": "Платные дополнительные занятия",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "no",
    },
    "activities": {
        "label": "Внеучебные занятия",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "yes",
    },
    "nursery": {
        "label": "Ходил в детский сад",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "yes",
    },
    "higher": {
        "label": "Хочет высшее образование",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "yes",
    },
    "internet": {
        "label": "Дома есть интернет",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "yes",
    },
    "romantic": {
        "label": "Романтические отношения",
        "help": "yes или no",
        "choices": ["yes", "no"],
        "default": "no",
    },
}


def example_payload() -> dict:
    payload = {name: spec["default"] for name, spec in NUMERIC_SPEC.items()}
    payload.update({name: spec["default"] for name, spec in CATEGORICAL_SPEC.items()})
    return payload
