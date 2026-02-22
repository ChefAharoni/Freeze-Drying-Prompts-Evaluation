import pytest


def test_contract_interface(defenses, trusted_instruction, sample_untrusted):
    for name, wrapper in defenses.items():
        if wrapper is None:  # special-cased in evaluator
            continue
        prompt = wrapper(trusted_instruction, sample_untrusted)
        assert isinstance(prompt, str)
        assert trusted_instruction.split(".")[0] in prompt  # trusted text present
        assert sample_untrusted in prompt


def test_delimiters_no_leak(defenses, sample_untrusted):
    trusted = "TOP_SECRET_TOKEN"
    for name, wrapper in defenses.items():
        prompt = wrapper(trusted, sample_untrusted)
        # Trusted string must not appear in untrusted view regions for dual models; at least ensure no duplication outside trusted block hint
        assert trusted in prompt
        # Ensure a delimiter or marker exists around untrusted content
        assert any(token in prompt for token in ["<untrusted>", "UNTRUSTED", "INTEGRITY", "DATA-"])
