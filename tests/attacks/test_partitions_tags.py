import json
from pathlib import Path


def test_attack_partitions_exist():
    base = Path(__file__).resolve().parents[2] / "data" / "attacks"
    expected = [
        "direct.jsonl",
        "roleplay.jsonl",
        "encoded.jsonl",
        "gradual.jsonl",
        "indirect.jsonl",
    ]
    for name in expected:
        assert (base / name).exists(), f"missing {name}; add tagged examples"


def test_records_have_attack_type():
    base = Path(__file__).resolve().parents[2] / "data" / "attacks"
    for path in base.glob("*.jsonl"):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                assert row.get("attack_type"), f"attack_type missing in {path}"
                assert row.get("text"), f"text missing in {path}"
