import numpy as np
import pandas as pd


class LowTurnoverMetaSimulator:
    """
    Realistic market friction simulator:
    - Lag-1 execution (signal at t-1 executed at t)
    - 15% Deadband rebalance filter to suppress chattering
    - T+2.5 minimum holding horizon (3 days)
    - 20 bps round-trip friction (15 bps base fee + 5 bps slippage)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        base_fee_bps: float = 15.0,
        slippage_bps: float = 5.0,
        rebalance_deadband: float = 0.15,
        min_hold_days: int = 3,
    ):
        self.initial_capital = initial_capital
        self.friction = (base_fee_bps + slippage_bps) / 10000.0
        self.rebalance_deadband = rebalance_deadband
        self.min_hold_days = min_hold_days

    def run_simulation(self, df_oos: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        aligned = df_oos[["bet_size"]].join(df_mkt[["close_adj"]]).dropna()
        aligned["daily_asset_return"] = aligned["close_adj"].pct_change().fillna(0.0)

        raw_weights = aligned["bet_size"].values
        n = len(raw_weights)
        smoothed_weights = np.zeros(n)

        current_w = 0.0
        days_held = 0

        for t in range(1, n):
            target_w = raw_weights[t - 1]

            if current_w > 0:
                days_held += 1

            if target_w == 0.0:
                if days_held >= self.min_hold_days or current_w == 0.0:
                    current_w = 0.0
                    days_held = 0
            else:
                if abs(target_w - current_w) >= self.rebalance_deadband:
                    current_w = target_w
                    if days_held == 0:
                        days_held = 1

            smoothed_weights[t] = current_w

        aligned["weight"] = smoothed_weights
        aligned["turnover"] = aligned["weight"].diff().abs().fillna(0.0)
        aligned["costs"] = aligned["turnover"] * self.friction
        aligned["gross_return"] = aligned["weight"] * aligned["daily_asset_return"]
        aligned["net_return"] = aligned["gross_return"] - aligned["costs"]
        aligned["equity"] = self.initial_capital * (1.0 + aligned["net_return"]).cumprod()
        aligned["benchmark_equity"] = self.initial_capital * (1.0 + aligned["daily_asset_return"]).cumprod()

        peak = aligned["equity"].cummax()
        aligned["drawdown"] = (aligned["equity"] - peak) / peak
        return aligned

    def print_tear_sheet(self, df_sim: pd.DataFrame):
        n_days = len(df_sim)
        total_ret = (df_sim["equity"].iloc[-1] / self.initial_capital) - 1.0
        ann_ret = (1.0 + total_ret) ** (252 / n_days) - 1.0
        vol_ann = df_sim["net_return"].std() * np.sqrt(252)
        sharpe = (df_sim["net_return"].mean() / (df_sim["net_return"].std() + 1e-9)) * np.sqrt(252)

        negative_returns = df_sim["net_return"][df_sim["net_return"] < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-9
        sortino = ann_ret / downside_std

        max_dd = df_sim["drawdown"].min()
        calmar = (ann_ret / abs(max_dd)) if abs(max_dd) > 0 else 0.0

        total_turnover = df_sim["turnover"].sum()
        total_costs = df_sim["costs"].sum()
        market_exposure = (df_sim["weight"] > 0).mean()

        bench_total_ret = (df_sim["benchmark_equity"].iloc[-1] / self.initial_capital) - 1.0
        bench_ann_ret = (1.0 + bench_total_ret) ** (252 / n_days) - 1.0
        bench_sharpe = (df_sim["daily_asset_return"].mean() / (df_sim["daily_asset_return"].std() + 1e-9)) * np.sqrt(252)

        print("\n" + "=" * 65)
        print("     HIỆU NĂNG MÔ PHỎNG LONG-ONLY META-LABELING (NET OF FEES)")
        print("=" * 65)
        print(f"Tổng số ngày quan sát       : {n_days} phiên")
        print(f"Tỷ lệ tham gia thị trường   : {market_exposure:.2%} (Thời gian cầm vị thế Mua)")
        print(f"Tổng khối lượng Turnover    : {total_turnover:.2f}x vốn")
        print(f"Tổng chi phí trượt & phí    : {total_costs:.2%}")
        print("-" * 65)
        print(f"Lợi nhuận lũy kế (Net)      : {total_ret:.2%}  |  Benchmark: {bench_total_ret:.2%}")
        print(f"Lợi nhuận hàng năm (CAGR)   : {ann_ret:.2%}   |  Benchmark: {bench_ann_ret:.2%}")
        print(f"Độ biến động hàng năm       : {vol_ann:.2%}")
        print(f"Sharpe Ratio (Net)          : {sharpe:.2f}    |  Benchmark: {bench_sharpe:.2f}")
        print(f"Sortino Ratio (Net)         : {sortino:.2f}")
        print(f"Calmar Ratio                : {calmar:.2f}")
        print(f"Max Drawdown (MDD)          : {max_dd:.2%}")
        print("=" * 65)


if __name__ == "__main__":
    import os
    if os.path.exists("df_oos_calibrated.csv") and os.path.exists("df_final.csv"):
        df_oos = pd.read_csv("df_oos_calibrated.csv", index_col=0, parse_dates=True)
        df_final = pd.read_csv("df_final.csv", index_col=0, parse_dates=True)
        sim = LowTurnoverMetaSimulator()
        df_sim = sim.run_simulation(df_oos, df_final)
        sim.print_tear_sheet(df_sim)