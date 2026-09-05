import numpy as np
import pandas as pd


class Layer0TripleBarrier:
    """
    Long-Only Dynamic Volatility-Scaled Triple Barrier Method.
    Label 1.0: Hits Upper Barrier (Take-Profit) before Lower Barrier and before time-out.
    Label 0.0: Hits Lower Barrier (Stop-Loss) or times out without hitting Upper Barrier.
    """

    @staticmethod
    def label(
        df: pd.DataFrame,
        h: int = 5,
        pt: float = 1.0,
        sl: float = 1.0,
        vol_span: int = 20
    ) -> pd.DataFrame:
        df = df.copy()
        log_ret = df["log_return"]

        # Local volatility dynamically scales the barriers
        sigma = log_ret.ewm(span=vol_span, min_periods=2).std().bfill()

        prices = df["close_adj"].values
        sigmas = sigma.values
        n = len(df)

        meta_target = np.full(n, np.nan)
        ret_barrier = np.full(n, np.nan)
        touch_time = np.full(n, np.nan, dtype=object)

        for i in range(n - h):
            p0 = prices[i]
            if pd.isna(p0) or pd.isna(sigmas[i]):
                continue

            upper_barrier = p0 * (1.0 + pt * sigmas[i])
            lower_barrier = p0 * (1.0 - sl * sigmas[i])

            window_prices = prices[i + 1 : i + h + 1]
            hit_upper = np.where(window_prices >= upper_barrier)[0]
            hit_lower = np.where(window_prices <= lower_barrier)[0]

            first_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
            first_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf

            if first_upper < first_lower and first_upper < np.inf:
                meta_target[i] = 1.0
                ret_barrier[i] = (window_prices[first_upper] / p0) - 1.0
                touch_time[i] = df.index[i + 1 + first_upper]
            else:
                meta_target[i] = 0.0
                if first_lower < first_upper and first_lower < np.inf:
                    ret_barrier[i] = (window_prices[first_lower] / p0) - 1.0
                    touch_time[i] = df.index[i + 1 + first_lower]
                else:
                    ret_barrier[i] = (window_prices[-1] / p0) - 1.0
                    touch_time[i] = df.index[i + h]

        df["local_volatility"] = sigma
        df["target_label"] = meta_target
        df["target_barrier_return"] = ret_barrier
        df["barrier_touch_time"] = touch_time
        return df