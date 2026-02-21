import math


def cost_metrics(TPR, FPR, p, C):
    f = TPR * p + FPR * (1 - p)
    M = 1 + f * (C - 1)
    S = TPR
    return f, M, S


def test_cost_monotonicity():
    p, C = 0.01, 5
    f1, M1, _ = cost_metrics(TPR=0.9, FPR=0.05, p=p, C=C)
    f2, M2, _ = cost_metrics(TPR=0.8, FPR=0.10, p=p, C=C)
    assert f1 < f2
    assert M1 < M2


def test_threshold_direction():
    # Lower thresholds should escalate more traffic (higher f) in this proxy
    p, C = 0.01, 3
    low_f, _, _ = cost_metrics(TPR=0.95, FPR=0.08, p=p, C=C)
    high_f, _, _ = cost_metrics(TPR=0.85, FPR=0.03, p=p, C=C)
    assert low_f > high_f
