import pytest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_asr", ROOT / "scripts" / "eval_asr.py")
eval_asr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_asr)  # type: ignore


@pytest.mark.stub
@pytest.mark.requires_data
def test_baseline_ordering(deepset_rows):
    results = eval_asr.evaluate_dataset("deepset", deepset_rows[:200])
    asr = {k: v["ASR"] for k, v in results.items()}
    assert asr["no_defense"] >= asr["prompt_hardening"]
    assert asr["no_defense"] >= asr["freeze_dry_standalone"]
