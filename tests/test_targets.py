from __future__ import annotations

import pandas as pd
import pytest

from app.labeling.targets import add_labels


def test_label_math():
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 3,
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [100.0, 110.0, 105.0],
        }
    )
    out = add_labels(df)
    assert out.loc[0, "fwd_ret_1d"] == pytest.approx(0.10)
    assert out.loc[0, "label_up_1d"] == 1.0
    assert out.loc[1, "fwd_ret_1d"] == pytest.approx(105 / 110 - 1)
    assert out.loc[1, "label_up_1d"] == 0.0
    assert pd.isna(out.loc[2, "label_up_1d"])


def test_threshold_filters_small_moves():
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 2,
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100.0, 100.5],
        }
    )
    assert add_labels(df, threshold=0.0).loc[0, "label_up_1d"] == 1.0
    assert add_labels(df, threshold=0.01).loc[0, "label_up_1d"] == 0.0
