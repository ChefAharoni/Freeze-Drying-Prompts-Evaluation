# Prompt Injection Defense Mini-Evaluation

Minimal, dependency-light suite to reproduce a small part of the paper's prompt-injection evaluation. It downloads two public Hugging Face datasets, wraps their prompts with a few baseline defenses, and measures attack success rate (ASR) and false-positive/refusal rate.

## Datasets
- `deepset/prompt-injections` (all 662 samples)
- `geekyrakshit/prompt-injection-dataset` (deterministic sample of ~10k rows; shuffled with seed=42 if larger)

ASR is computed only on rows labeled **INJECTION**. Rows labeled **LEGIT** contribute to the false-positive/refusal rate (the rate at which a defense refuses or produces non-summary output for benign text).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download datasets
```bash
python scripts/download_datasets.py
```
This writes normalized JSONL files to `data/deepset.jsonl` and `data/combined.jsonl` with fields: `{id, dataset, split, text, label}`.

## Run evaluations
```bash
python scripts/eval_asr.py --dataset deepset --limit 662
python scripts/eval_asr.py --dataset combined --limit 10000
```

If `OPENAI_API_KEY` is set (and optionally `OPENAI_MODEL`, default `gpt-4o-mini`), the script will call the OpenAI Chat Completions API. Otherwise it uses a deterministic stub so the suite runs offline and still produces `results/results.json`.

## Defenses implemented
- `no_defense`
- `xml_delimiters` (wrap untrusted text in `<untrusted>` tags)
- `prompt_hardening` (defensive instruction + delimiters)
- `freeze_dry_standalone` (placeholder `protect()` you can replace with the paper's freeze-dry logic)

## Outputs
- Console summary with ASR and benign refusal rate per defense.
- `results/results.json` containing metrics, average prompt length, and overhead vs. no defense (character-length proxy).
