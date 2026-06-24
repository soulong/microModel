import json
import re

import yaml

_NUMERIC_RE = re.compile(r'^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$')


def _coerce_numeric(value):
    if isinstance(value, str) and _NUMERIC_RE.match(value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    return value


_REQUIRED_PATHS = [
    ["data", "train_dir"],
    ["model", "backbone"],
    ["model", "channel_indices"],
    ["model", "target_size"],
    ["training", "batch_size"],
    ["training", "epochs"],
    ["training", "lr"],
    ["training", "weight_decay"],
    ["output", "run_dir"],
]


class Config(dict):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        text = text.replace("\\", "/")
        super().__init__(_coerce_numeric(yaml.safe_load(text)))
        self._validate()

    def _validate(self):
        missing = []
        for keys in _REQUIRED_PATHS:
            section = self
            for k in keys:
                if not isinstance(section, dict) or k not in section:
                    missing.append(".".join(keys))
                    break
                section = section[k]
        if missing:
            raise KeyError(
                f"Config missing required key(s): {', '.join(missing)}. "
                f"Check your YAML file for typos or missing sections."
            )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(dict(self), f)


class TrainingLogger:
    def __init__(self, path):
        self.path = path

    def log(self, entry):
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
