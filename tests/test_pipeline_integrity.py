import unittest
import numpy as np
import pandas as pd
from src.data_integrity import Layer1DataIntegrity, Layer2MicrostructureTopologies
from src.labeling import Layer0TripleBarrier
from src.calibration import StandardPurgedCV, extract_primary_signals
from src.simulator import LowTurnoverMetaSimulator


class TestPipelineIntegrity(unittest.TestCase):

    def setUp(self):
        # Tạo chuỗi OHLCV giả lập có tính ngẫu nhiên nhưng xác định
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=150, freq="B")
        close = 100.0 * np.cumprod(1 + np.random.normal(0, 0.02, size=150))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, size=150)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, size=150)))
        open_ = (high + low) / 2.0
        volume = np.random.randint(100000, 5000000, size=150)

        self.df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    def test_layer1_geometric_integrity(self):
        df_clean, stats = Layer1DataIntegrity.run_audit(self.df)
        self.assertGreaterEqual(stats["clean_bars"], 100)
        self.assertTrue((df_clean["high_adj"] >= df_clean["low_adj"]).all())
        self.assertTrue((df_clean["low_adj"] > 0).all())

    def test_triple_barrier_labeling(self):
        df_clean, _ = Layer1DataIntegrity.run_audit(self.df)
        df_topo = Layer2MicrostructureTopologies.compute(df_clean)
        df_labeled = Layer0TripleBarrier.label(df_topo, h=5, pt=1.0, sl=1.0, vol_span=20)

        self.assertIn("target_label", df_labeled.columns)
        valid_labels = df_labeled["target_label"].dropna().unique()
        for lab in valid_labels:
            self.assertIn(lab, [0.0, 1.0])

    def test_purged_cv_leakage(self):
        purge_gap = 5
        cv = StandardPurgedCV(n_splits=3, purge_gap=purge_gap)
        for train_idx, test_idx in cv.split(self.df):
            # Điểm cuối cùng của tập train phải cách điểm đầu tập test ít nhất một khoảng purge_gap
            self.assertLessEqual(train_idx[-1], test_idx[0] - purge_gap)

    def test_simulator_deadband_and_latency(self):
        sim = LowTurnoverMetaSimulator(
            initial_capital=100000,
            rebalance_deadband=0.15,
            min_hold_days=3
        )
        dates = self.df.index[-50:]
        df_oos = pd.DataFrame(
            {"bet_size": [0.0] * 10 + [0.5] * 20 + [0.0] * 20},
            index=dates
        )
        df_mkt = pd.DataFrame({"close_adj": self.df["close"].loc[dates]}, index=dates)
        sim_res = sim.run_simulation(df_oos, df_mkt)

        # Kiểm tra chi phí và turnover được tính toán hợp lệ
        self.assertTrue((sim_res["turnover"] >= 0).all())
        self.assertTrue((sim_res["costs"] >= 0).all())
        self.assertEqual(len(sim_res), 50)


if __name__ == "__main__":
    unittest.main()