import numpy as np
import pandas as pd


class Layer1DataIntegrity:
    """
    Stage 1: Deduplication, geometric bar consistency checks,
    forward-adjustment protocol, and source-level zero-variance dropping.
    """

    @staticmethod
    def run_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        initial_len = len(df)

        # 1. Deduplication & Chronological Sort
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # 2. Geometric bounds audit
        max_oc = df[["open", "close"]].max(axis=1)
        min_oc = df[["open", "close"]].min(axis=1)
        df["high"] = np.maximum(df["high"], max_oc)
        df["low"] = np.minimum(df["low"], min_oc)
        df = df[(df["low"] > 0) & (df["volume"] >= 0)]

        # 3. Forward-Adjustment Protocol
        if "split_factor" not in df.columns:
            df["split_factor"] = 1.0
        if "dividend" not in df.columns:
            df["dividend"] = 0.0

        daily_ret = ((df["close"] + df["dividend"]) / (df["close"].shift(1) * df["split_factor"])) - 1.0
        df["close_adj"] = df["close"].iloc[0] * (1.0 + daily_ret).cumprod()
        df["close_adj"] = df["close_adj"].fillna(df["close"])

        ratio = df["close_adj"] / df["close"]
        for col in ["open", "high", "low"]:
            df[f"{col}_adj"] = df[col] * ratio

        # 4. Zero-Variance Drop
        dropped_zero_var = []
        for col in df.columns:
            if col not in ["split_factor", "dividend", "target_label"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].std() < 1e-8 or df[col].nunique() <= 1:
                        dropped_zero_var.append(col)

        if dropped_zero_var:
            df.drop(columns=dropped_zero_var, inplace=True)

        return df, {
            "initial_bars": initial_len,
            "clean_bars": len(df),
            "dropped_flat_cols": dropped_zero_var,
        }


class Layer2MicrostructureTopologies:
    """
    Standard OHLCV differential topologies & Level 2 multi-depth order book parser.
    """

    @staticmethod
    def compute(df: pd.DataFrame, max_depth_levels: int = 10) -> pd.DataFrame:
        df = df.copy()

        # 1. Standard Differential Topologies
        df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
        df["overnight_return"] = np.log(df["open_adj"] / df["close_adj"].shift(1))
        df["intraday_return"] = np.log(df["close_adj"] / df["open_adj"])
        df["hl_log_range"] = np.log(df["high_adj"] / df["low_adj"])

        # 2. L2 Multi-Level Order Book scan
        available_levels = [
            i for i in range(1, max_depth_levels + 1)
            if f"bid_price_{i}" in df.columns and f"ask_price_{i}" in df.columns
        ]

        if len(available_levels) > 1:
            df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
            df["l2_spread"] = df["ask_price_1"] - df["bid_price_1"]

            total_weighted_bid_size = np.zeros(len(df))
            total_weighted_ask_size = np.zeros(len(df))
            cross_weighted_bid_price = np.zeros(len(df))
            cross_weighted_ask_price = np.zeros(len(df))

            for i in available_levels:
                decay_weight = np.exp(-(i - 1) * 0.5)
                b_size_w = df[f"bid_size_{i}"] * decay_weight
                a_size_w = df[f"ask_size_{i}"] * decay_weight

                total_weighted_bid_size += b_size_w
                total_weighted_ask_size += a_size_w
                cross_weighted_bid_price += df[f"bid_price_{i}"] * b_size_w
                cross_weighted_ask_price += df[f"ask_price_{i}"] * a_size_w

            imbalance_denominator = total_weighted_bid_size + total_weighted_ask_size + 1e-8
            df["order_imbalance_depth"] = (total_weighted_bid_size - total_weighted_ask_size) / imbalance_denominator
            df["micro_price_depth"] = (
                (cross_weighted_bid_price / (total_weighted_bid_size + 1e-8)) * total_weighted_ask_size
                + (cross_weighted_ask_price / (total_weighted_ask_size + 1e-8)) * total_weighted_bid_size
            ) / imbalance_denominator
            df["micro_mid_deviation"] = df["micro_price_depth"] - df["mid_price"]

        elif all(col in df.columns for col in ["best_bid", "best_ask", "bid_size", "ask_size"]):
            df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2.0
            imbalance_denominator = df["bid_size"] + df["ask_size"] + 1e-8
            df["micro_price"] = (df["best_bid"] * df["ask_size"] + df["best_ask"] * df["bid_size"]) / imbalance_denominator
            df["l2_spread"] = df["best_ask"] - df["best_bid"]
            df["micro_mid_deviation"] = df["micro_price"] - df["mid_price"]
            df["order_imbalance_depth"] = (df["bid_size"] - df["ask_size"]) / imbalance_denominator

        return df