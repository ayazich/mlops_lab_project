from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.schema import TARGET


def prepare(params_path="params.yaml"):
    cfg = yaml.safe_load(Path(params_path).read_text())["data"]
    source = Path(cfg["source"])
    if not source.is_file():
        raise FileNotFoundError(
            f"Не найден {source}. Положите student-por.csv в data/ "
            "(разделитель — точка с запятой)."
        )
    frame = pd.read_csv(source, sep=cfg.get("separator", ";"))
    if TARGET not in frame.columns:
        raise ValueError(f"В {source} нет столбца {TARGET}")
    train, test = train_test_split(
        frame,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
    )
    Path("data").mkdir(exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"Prepared {len(train)} train / {len(test)} test examples from {source}")


if __name__ == "__main__":
    prepare()
