import pytest


@pytest.mark.stub
def test_threshold_monotonicity():
    # simple proxy: higher threshold -> fewer escalations; here we simulate counts
    thresholds = [0.2, 0.5, 0.8]
    traffic = [100, 100, 100]
    escalations = []
    for t, n in zip(thresholds, traffic):
        escalations.append(int(n * (1 - t)))
    assert escalations[0] >= escalations[1] >= escalations[2]
