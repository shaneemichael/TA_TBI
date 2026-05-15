import json
from pathlib import Path

import pytest

from nya_ir.experiment import RetrieverName, StrategyName
from nya_ir.utils.config import load_config
from nya_ir.utils.hashing import stable_hash

ROOT = Path(__file__).resolve().parents[1]


def test_load_json_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 42, "strategy": "keep"}), encoding="utf-8")
    assert load_config(path) == {"seed": 42, "strategy": "keep"}


def test_load_config_rejects_non_mapping_top_level(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(["keep"]), encoding="utf-8")

    with pytest.raises(ValueError, match="Config must contain a top-level object"):
        load_config(path)


def test_stable_hash_is_order_independent():
    left = stable_hash({"a": 1, "b": 2})
    right = stable_hash({"b": 2, "a": 1})
    assert left == right


def test_experiment_matrix_matches_canonical_enums():
    config = load_config(ROOT / "configs" / "experiment_matrix.yaml")

    assert config["strategies"] == [strategy.value for strategy in StrategyName]
    assert config["retrievers"] == [retriever.value for retriever in RetrieverName]


def test_strategy_config_entries_match_canonical_enums():
    config = load_config(ROOT / "configs" / "strategies.yaml")
    entries = config["strategies"]

    assert [entry["name"] for entry in entries] == [strategy.value for strategy in StrategyName]
    assert all("description" in entry for entry in entries)


def test_retriever_config_keys_match_canonical_enums():
    config = load_config(ROOT / "configs" / "retrievers.yaml")

    assert list(config["retrievers"]) == [retriever.value for retriever in RetrieverName]
