import json

from nya_ir.utils.config import load_config
from nya_ir.utils.hashing import stable_hash


def test_load_json_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 42, "strategy": "keep"}), encoding="utf-8")
    assert load_config(path) == {"seed": 42, "strategy": "keep"}


def test_stable_hash_is_order_independent():
    left = stable_hash({"a": 1, "b": 2})
    right = stable_hash({"b": 2, "a": 1})
    assert left == right

