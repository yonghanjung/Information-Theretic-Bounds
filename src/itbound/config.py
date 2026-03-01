from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from fbound.utils.data_generating import generate_data


class ConfigError(ValueError):
    """Configuration error for itbound CLI."""


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ConfigError("Config must be YAML (.yml/.yaml) or JSON (.json).")

    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping (YAML/JSON object).")

    return validate_config(data)


def _require_keys(d: Dict[str, Any], keys: Tuple[str, ...], ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"Missing required keys for {ctx}: {missing}")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a dict.")

    _require_keys(
        cfg,
        (
            "data",
            "divergence",
            "propensity_model",
            "m_model",
            "dual_net_config",
            "fit_config",
            "seed",
        ),
        "root",
    )

    out = dict(cfg)
    out.setdefault("phi", "identity")
    out.setdefault("output_path", "itbound_bounds.csv")

    data = out.get("data")
    if not isinstance(data, dict):
        raise ConfigError("data must be a mapping.")

    if "synthetic" in data:
        synthetic = data["synthetic"]
        if not isinstance(synthetic, dict):
            raise ConfigError("data.synthetic must be a mapping.")
        _require_keys(synthetic, ("n", "d"), "data.synthetic")
        synthetic.setdefault("seed", out["seed"])
        synthetic.setdefault("structural_type", "linear")
    elif "npz_path" in data:
        if not isinstance(data["npz_path"], str):
            raise ConfigError("data.npz_path must be a string.")
    elif "csv_path" in data:
        _require_keys(data, ("csv_path", "y_col", "a_col", "x_cols"), "data.csv")
        if not isinstance(data["x_cols"], list):
            raise ConfigError("data.x_cols must be a list of column names.")
    else:
        raise ConfigError("data must specify one of: synthetic, npz_path, csv_path.")

    return out


def build_phi(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = str(name).strip().lower()
    if key == "identity":
        return lambda y: y
    raise ConfigError(f"Unsupported phi: {name}. Supported: identity.")


def resolve_data(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Callable]]:
    data = cfg["data"]
    if "synthetic" in data:
        synthetic = data["synthetic"]
        gen = generate_data(
            n=int(synthetic["n"]),
            d=int(synthetic["d"]),
            seed=int(synthetic["seed"]),
            structural_type=str(synthetic.get("structural_type", "linear")),
        )
        return gen["Y"], gen["A"], gen["X"], gen.get("GroundTruth")

    if "npz_path" in data:
        payload = np.load(data["npz_path"])
        return payload["Y"], payload["A"], payload["X"], None

    frame = pd.read_csv(data["csv_path"])
    y = frame[data["y_col"]].to_numpy()
    a = frame[data["a_col"]].to_numpy()
    x = frame[data["x_cols"]].to_numpy()
    return y, a, x, None


def default_example_config(output_path: Path) -> Dict[str, Any]:
    return {
        "data": {
            "synthetic": {
                "n": 50,
                "d": 3,
                "seed": 123,
                "structural_type": "linear",
            }
        },
        "divergence": "KL",
        "phi": "identity",
        "propensity_model": "logistic",
        "m_model": "linear",
        "dual_net_config": {
            "hidden_sizes": [16, 16],
            "activation": "relu",
            "dropout": 0.0,
            "h_clip": 10.0,
            "device": "cpu",
        },
        "fit_config": {
            "n_folds": 2,
            "num_epochs": 2,
            "batch_size": 16,
            "lr": 5e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 5.0,
            "eps_propensity": 1e-3,
            "deterministic_torch": True,
            "train_m_on_fold": True,
            "propensity_config": {
                "C": 1.0,
                "max_iter": 200,
                "penalty": "l2",
                "solver": "lbfgs",
                "n_jobs": 1,
            },
            "m_config": {"alpha": 1.0},
            "verbose": False,
            "log_every": 1,
        },
        "seed": 123,
        "output_path": str(output_path),
    }
