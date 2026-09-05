import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class StandardPurgedCV:
    """
    Purged & Embargoed TimeSeriesSplit to eliminate horizon target leakage.
    """

    def __init__(self, n_splits=5, purge_gap=5, max_train_size=None):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
        for tr_idx, te_idx in tscv.split(X):
            tr_purged = tr_idx[tr_idx < (te_idx[0] - self.purge_gap)]
            if len(tr_purged) > 0 and len(te_idx) > 0:
                yield tr_purged, te_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def optimize_meta_threshold(oof_probs, oof_true, min_trades=20, min_coverage=0.15, beta=0.5):
    """
    Risk-adjusted Utility Optimization on OOF probability vectors using F0.5 metric.
    Bounded strictly in [0.480, 0.540] floor to maintain production safety.
    """
    best_score = -np.inf
    best_threshold = None
    n_samples = len(oof_probs)
    grid = np.linspace(0.45, 0.58, 27)

    for th in grid:
        preds = (oof_probs >= th).astype(int)
        n_trades = np.sum(preds)
        if n_trades < min_trades:
            continue

        tp = np.sum((preds == 1) & (oof_true == 1))
        fp = np.sum((preds == 1) & (oof_true == 0))
        fn = np.sum((preds == 0) & (oof_true == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall == 0:
            continue

        beta_sq = beta**2
        f_score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-8)
        coverage = n_trades / n_samples
        penalty = 1.0 if coverage >= min_coverage else (coverage / min_coverage)
        utility_score = f_score * penalty

        if utility_score > best_score:
            best_score = utility_score
            best_threshold = th

    if best_threshold is None:
        best_threshold = float(np.percentile(oof_probs, 60))
        best_score = 0.50

    bounded_threshold = float(np.clip(best_threshold, 0.480, 0.540))
    return bounded_threshold, best_score


def compute_meta_confidence_sizing(calibrated_probs, meta_thresh, base_min_weight=0.25):
    """
    Non-linear position scaling bounded by entry safety floor [base_min_weight, 1.0].
    """
    excess_prob = np.maximum(calibrated_probs - meta_thresh, 0.0)
    max_excess = 1.0 - meta_thresh + 1e-8
    scaled_weight = base_min_weight + (1.0 - base_min_weight) * (excess_prob / max_excess)
    return np.where(calibrated_probs >= meta_thresh, np.clip(scaled_weight, base_min_weight, 1.0), 0.0)


def compute_fractional_kelly_sizing(calibrated_probs, meta_thresh, b_ratio=1.0, half_kelly=0.5):
    """
    Fractional Kelly criterion proxy.
    """
    kelly = (calibrated_probs * (b_ratio + 1.0) - 1.0) / (b_ratio + 1e-8)
    return np.where(calibrated_probs >= meta_thresh, np.clip(kelly * half_kelly, 0.05, 1.0), 0.0)


def extract_primary_signals(df: pd.DataFrame) -> pd.Series:
    """
    Primary Model Filter: Long-only opportunities under macro trend support.
    Shifted by 1 session to prevent lookahead bias.
    """
    c = df["close_adj"]
    ema_50 = c.ewm(span=50, adjust=False).mean()
    trend_filter = c > ema_50

    if "mkt_return" in df.columns:
        mkt_ma = df["mkt_return"].rolling(20).mean()
        mkt_filter = mkt_ma > -0.006
    else:
        mkt_filter = pd.Series(True, index=df.index)

    return (trend_filter & mkt_filter).shift(1).fillna(0).astype(int)