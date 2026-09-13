
import numpy as np
import pandas as pd

RFM_FEATURE_VERSION = "rfm-v3"


def score_rfm_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    ranks = series.rank(method="average", pct=True, ascending=higher_is_better)
    return np.clip(np.ceil(ranks * 5), 1, 5).astype(int)


def assign_rfm_segment(r_score: int, f_score: int, m_score: int) -> str:
    r, f, m = int(r_score), int(f_score), int(m_score)

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r <= 2 and f >= 3 and m >= 3:
        return "Need Attention"
    if r <= 2 and f >= 2 and m >= 2:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    if r >= 4 and f >= 3:
        return "Potential Loyalists"
    if r >= 4 and f <= 2:
        return "New Customers"
    if f >= 4 and m >= 3:
        return "Loyal Customers"
    if m >= 4:
        return "Big Spenders"
    if f >= 3 and r >= 3:
        return "Regular Customers"
    return "Lost"
