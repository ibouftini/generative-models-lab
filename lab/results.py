"""Results files: the source of every number quoted in the course.

    results.save("chapter3", "ring", {"high_quality": 0.615}, config={...}, seed=0, script=__file__)

writes results/chapter3/ring.json with the numbers, the configuration, the seed, the script that produced them, the
library versions, the device and the date. results.load("chapter3", "ring") reads it back.
"""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import RESULTS, ROOT


def _environment():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def save(chapter, name, numbers, config=None, seed=None, script=None):
    path = RESULTS / chapter / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"numbers": numbers, "config": config or {}, "seed": seed,
              "script": Path(script).resolve().relative_to(ROOT).as_posix() if script else None,
              "environment": _environment()}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def load(chapter, name):
    return json.loads((RESULTS / chapter / f"{name}.json").read_text(encoding="utf-8"))
