import numpy as np
import pandas as pd
import pytest

from tounum_cls import evaluation, modeling


def test_evaluate_binary_values():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.2, 0.7, 0.8, 0.1])
    r = evaluation.evaluate_binary(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 2)
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert 0 <= r["roc_auc"] <= 1
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_evaluate_binary_single_class_auc_nan():
    r = evaluation.evaluate_binary([0, 0], [0.1, 0.2])
    assert np.isnan(r["roc_auc"]) and r["accuracy"] == 1.0


def test_per_source_accuracy():
    meta = pd.DataFrame({"source": ["Photo", "Photo", "Text", "Text"], "label": [1, 1, 0, 0]})
    out = evaluation.per_source_accuracy(meta, [0.9, 0.2, 0.1, 0.1])
    acc = dict(zip(out["source"], out["accuracy"]))
    assert acc == {"Photo": 0.5, "Text": 1.0}
    assert out.iloc[0]["source"] == "Photo"  # trié du pire au meilleur


def _hist(train_acc, val_acc, val_loss):
    n = len(train_acc)
    return pd.DataFrame({"epoch": range(1, n + 1), "accuracy": train_acc, "val_accuracy": val_acc,
                         "loss": [0.1] * n, "val_loss": val_loss})


def test_diagnose_fit_three_regimes():
    under = evaluation.diagnose_fit(_hist([0.6, 0.7], [0.6, 0.7], [0.7, 0.6]))
    over = evaluation.diagnose_fit(_hist([0.9, 0.99], [0.9, 0.85], [0.3, 0.5]))
    good = evaluation.diagnose_fit(_hist([0.9, 0.97], [0.9, 0.96], [0.3, 0.2]))
    assert "biais" in under["verdict"]
    assert "variance" in over["verdict"] and over["best_epoch"] == 1 and over["gap_acc"] == pytest.approx(0.14)
    assert "correct" in good["verdict"] and good["best_epoch"] == 2
