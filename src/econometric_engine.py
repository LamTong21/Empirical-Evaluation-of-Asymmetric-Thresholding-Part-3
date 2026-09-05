import math
import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller, bds, kpss


# =====================================================================
# DIAGNOSTIC SENSORS (STAGE 2)
# =====================================================================

class Layer4MemoryDependence:
    @staticmethod
    def compute_hurst_dfa(series: pd.Series) -> float:
        s = series.dropna().values
        y = np.cumsum(s - np.mean(s))
        n = len(y)
        if n < 40:
            return 0.5

        scales = np.floor(np.logspace(np.log10(10), np.log10(n // 4), num=15)).astype(int)
        scales = np.unique(scales)
        fluctuations = []

        for s_len in scales:
            num_segments = n // s_len
            segment_fluct = []
            for i in range(num_segments):
                seg = y[i * s_len : (i + 1) * s_len]
                x = np.arange(s_len)
                poly = np.polyfit(x, seg, 1)
                trend = np.polyval(poly, x)
                segment_fluct.append(np.sqrt(np.mean((seg - trend) ** 2)))
            fluctuations.append(np.mean(segment_fluct))

        poly_hurst = np.polyfit(np.log(scales), np.log(fluctuations), 1)
        return float(poly_hurst[0])

    @staticmethod
    def lo_mackinlay_vr(prices: pd.Series, k: int) -> dict:
        p = np.log(prices.dropna().values)
        t = len(p)
        if t <= k + 2:
            return {"vr": 1.0, "z_stat": 0.0, "p_value": 1.0}

        r1 = p[1:] - p[:-1]
        mu = (p[-1] - p[0]) / (t - 1)
        var_1 = np.sum((r1 - mu) ** 2) / (t - 2)

        rk = p[k:] - p[:-k]
        m = k * (t - k) * (1 - (k / (t - 1)))
        var_k = np.sum((rk - k * mu) ** 2) / m
        vr = var_k / var_1

        delta = np.zeros(k - 1)
        denom = (np.sum((r1 - mu) ** 2)) ** 2
        for j in range(1, k):
            num = np.sum(((r1[j:] - mu) ** 2) * ((r1[:-j] - mu) ** 2))
            delta[j - 1] = ((2.0 * (k - j) / k) ** 2) * (num / denom)

        phi_k = np.sum(delta)
        z_star = (vr - 1.0) / np.sqrt(phi_k) if phi_k > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_star)))
        return {"vr": float(vr), "z_stat": float(z_star), "p_value": float(p_val)}

    @classmethod
    def multi_scale_memory(cls, prices: pd.Series, returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 50:
            return {
                "dynamic_lags": [1, 3, 5],
                "hurst": 0.5,
                "short_term_mean_reverting": False,
                "long_term_trending": False,
            }

        max_lag = min(30, len(r) // 3)
        acf_vals = acf(r, nlags=max_lag, fft=True)
        conf_interval = 1.96 / np.sqrt(len(r))
        sig_lags = [i for i, val in enumerate(acf_vals[1:], 1) if abs(val) > conf_interval]
        if not sig_lags:
            sig_lags = [1, 3, 5]

        vr_short = cls.lo_mackinlay_vr(prices, k=3)
        vr_long = cls.lo_mackinlay_vr(prices, k=20)

        return {
            "dynamic_lags": sorted(list(set(sig_lags[:5] + [1, 5, 20]))),
            "hurst": cls.compute_hurst_dfa(returns),
            "short_term_mean_reverting": bool(vr_short["vr"] < 1.0 and vr_short["p_value"] < 0.05),
            "long_term_trending": bool(vr_long["vr"] > 1.0 and vr_long["p_value"] < 0.05),
        }


class Layer4bSpectralCycleDiagnostics:
    @staticmethod
    def run(prices: pd.Series) -> dict:
        p = prices.dropna().values
        n = len(p)
        if n < 40:
            return {"dominant_cycle_len": 10}

        detrended = p - np.polyval(np.polyfit(np.arange(n), p, 1), np.arange(n))
        fft_vals = np.abs(rfft(detrended))
        freqs = rfftfreq(n, d=1.0)
        fft_vals[0] = 0
        peak_idx = np.argmax(fft_vals)
        dominant_freq = freqs[peak_idx] if freqs[peak_idx] > 0 else 0.1
        dominant_cycle = int(np.clip(1.0 / dominant_freq, 3, 30))
        return {"dominant_cycle_len": dominant_cycle}


class Layer5FractionalIntegration:
    @staticmethod
    def get_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w[::-1])

    @classmethod
    def fractionally_diff(cls, series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
        weights = cls.get_weights(d, len(series), threshold)
        width = len(weights)
        vals = series.values
        res = [np.dot(weights, vals[i - width : i]) for i in range(width, len(vals))]
        return pd.Series(res, index=series.index[width:], name=f"frac_diff_{d:.2f}")

    @classmethod
    def find_optimal_d(cls, series: pd.Series, d_step: float = 0.05) -> dict:
        s = series.dropna()
        if len(s) < 50:
            return {"optimal_d": 1.0}

        best_d = 1.0
        for d in np.arange(0.0, 1.05, d_step):
            fd = cls.fractionally_diff(s, d)
            if len(fd) < 50:
                continue
            adf_p = adfuller(fd.dropna(), autolag="AIC")[1]
            if adf_p < 0.05:
                best_d = float(round(d, 2))
                break
        return {"optimal_d": best_d}


class Layer6VolatilityDynamics:
    @staticmethod
    def estimate(df: pd.DataFrame) -> dict:
        c = df["close_adj"]
        ret = np.log(c / c.shift(1)).dropna()
        if len(ret) < 30:
            return {"has_arch_effect": False, "has_asymmetric_vol": False}

        lm_stat, lm_p, _, _ = het_arch(ret, nlags=5)
        ret_lag = ret.shift(1).dropna()
        vol_proxy = (ret**2).iloc[1:]
        aligned_df = pd.concat([ret_lag, vol_proxy], axis=1).dropna()
        leverage_corr = aligned_df.iloc[:, 0].corr(aligned_df.iloc[:, 1])

        return {
            "has_arch_effect": bool(lm_p < 0.05),
            "has_asymmetric_vol": bool(leverage_corr < -0.1),
        }


class Layer6bVolatilityJumpDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame, window: int = 20) -> dict:
        r = df["log_return"].dropna()
        if len(r) < window * 2:
            return {"has_volatility_jumps": False}
        rv = (r**2).rolling(window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(window).sum()
        jump_ratio = np.maximum(rv - bv, 0) / (rv + 1e-8)
        significant_jumps = (jump_ratio > 0.25).sum()
        return {"has_volatility_jumps": bool(significant_jumps > (len(r) * 0.05))}


class Layer7bVolumeDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        v = df["volume"].dropna()
        r_abs = np.log(df["close_adj"] / df["close_adj"].shift(1)).abs().dropna()
        aligned = pd.concat([v, r_abs], axis=1).dropna()
        if aligned.empty or len(aligned) < 30:
            return {"is_volume_significant": False}
        corr, p_val = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return {"is_volume_significant": bool(p_val < 0.05 and corr > 0.1)}


class Layer7cMicrostructureDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        r = df["log_return"].fillna(0)
        if "micro_mid_deviation" not in df.columns or "l2_spread" not in df.columns:
            return {"has_l2_orderbook": False, "micro_deviation_sig": False}

        micro_dev = df["micro_mid_deviation"].fillna(0)
        dev_corr, dev_pval = spearmanr(micro_dev.shift(1).fillna(0), r)
        return {
            "has_l2_orderbook": True,
            "micro_deviation_sig": bool(dev_pval < 0.05 and abs(dev_corr) > 0.05),
        }


class Layer8NonlinearDependence:
    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 50:
            return {"is_nonlinear_dependent": False}
        ar_res = AutoReg(r.values, lags=1).fit()
        resid = ar_res.resid
        resid_std = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-8)
        bds_stat, p_val = bds(resid_std, max_dim=2, epsilon=None)
        return {"is_nonlinear_dependent": bool(p_val < 0.05)}


class Layer8bComplexityDiagnostics:
    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna().values
        if len(r) < 30:
            return {"is_high_complexity": False}
        m, tau = 3, 1
        n = len(r) - (m - 1) * tau
        patterns = np.array([r[i : i + m * tau : tau] for i in range(n)])
        ranks = np.argsort(patterns, axis=1)
        _, counts = np.unique(ranks, axis=0, return_counts=True)
        probs = counts / counts.sum()
        pe = -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(math.factorial(m))
        return {"is_high_complexity": bool(pe > 0.85)}


class Stage2SingleAssetDGPScanner:
    @staticmethod
    def execute(df: pd.DataFrame) -> dict:
        returns = df["log_return"].dropna()
        prices = df["close_adj"].dropna()

        mem_stats = Layer4MemoryDependence.multi_scale_memory(prices, returns)
        cycle_stats = Layer4bSpectralCycleDiagnostics.run(prices)
        frac_stats = Layer5FractionalIntegration.find_optimal_d(prices)
        vol_stats = Layer6VolatilityDynamics.estimate(df)
        jump_stats = Layer6bVolatilityJumpDiagnostics.run(df)
        volu_stats = Layer7bVolumeDiagnostics.run(df)
        nonlin_stats = Layer8NonlinearDependence.run(returns)
        complex_stats = Layer8bComplexityDiagnostics.run(returns)
        micro_stats = Layer7cMicrostructureDiagnostics.run(df)

        return {
            "dynamic_lags": mem_stats["dynamic_lags"],
            "optimal_d": frac_stats["optimal_d"],
            "dominant_cycle": cycle_stats["dominant_cycle_len"],
            "flags": {
                "has_long_trend": mem_stats["long_term_trending"],
                "has_short_reversion": mem_stats["short_term_mean_reverting"],
                "has_vol_clustering": vol_stats["has_arch_effect"],
                "has_asymmetric_vol": vol_stats["has_asymmetric_vol"],
                "is_volume_significant": volu_stats["is_volume_significant"],
                "is_nonlinear": nonlin_stats["is_nonlinear_dependent"],
                "has_vol_jumps": jump_stats["has_volatility_jumps"],
                "is_high_complexity": complex_stats["is_high_complexity"],
                "has_l2_orderbook": micro_stats["has_l2_orderbook"],
                "micro_deviation_sig": micro_stats["micro_deviation_sig"],
            },
        }


# =====================================================================
# FEATURE STRATEGIES & ROUTER (STAGE 3)
# =====================================================================

class BaseFeatureStrategy(ABC):
    @abstractmethod
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        pass


class BaselineStrategy(BaseFeatureStrategy):
    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]
        candle_range = (h - l) + eps

        feat["geo_body_ratio"] = (c - o) / candle_range
        feat["geo_upper_shadow"] = (h - np.maximum(o, c)) / candle_range
        feat["geo_lower_shadow"] = (np.minimum(o, c) - l) / candle_range
        feat["overnight_gap"] = (o / c.shift(1)) - 1.0

        log_v = np.log(v + eps)
        for k in self.lags:
            if k >= 3:
                roll_vol_mean = v.shift(1).rolling(k).mean()
                feat[f"rvol_{k}"] = v / (roll_vol_mean + eps)
                mu_lv = log_v.shift(1).rolling(k).mean()
                std_lv = log_v.shift(1).rolling(k).std(ddof=1)
                feat[f"vol_shock_{k}"] = (log_v - mu_lv) / (std_lv + eps)

        if isinstance(df.index, pd.DatetimeIndex):
            dow = df.index.dayofweek
            month = df.index.month
            feat["sin_dow"] = np.sin(2 * np.pi * dow / 5.0)
            feat["cos_dow"] = np.cos(2 * np.pi * dow / 5.0)
            feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
            feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)
        return feat


class TrendMomentumStrategy(BaseFeatureStrategy):
    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        for k in self.lags:
            ret_k = np.log(p / p.shift(k))
            feat[f"ret_{k}"] = ret_k
            feat[f"tsmom_sign_{k}"] = np.sign(ret_k)
            if k >= 3:
                roll_max = p.shift(1).rolling(k).max()
                roll_min = p.shift(1).rolling(k).min()
                feat[f"breakout_ratio_{k}"] = (p - roll_max) / ((roll_max - roll_min) + eps)

        ema_12 = p.ewm(span=12, adjust=False).mean()
        ema_26 = p.ewm(span=26, adjust=False).mean()
        feat["macd_dist"] = (ema_12 / ema_26) - 1.0
        feat["macd_slope_5"] = feat["macd_dist"] - feat["macd_dist"].shift(5)
        return feat


class OscillatorMeanReversionStrategy(BaseFeatureStrategy):
    def __init__(self, dynamic_lags: list[int]):
        self.lags = [k for k in dynamic_lags if k <= 20] or [5, 10, 20]

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        for k in self.lags:
            if k >= 3:
                roll_mean = p.shift(1).rolling(k).mean()
                roll_std = p.shift(1).rolling(k).std(ddof=1)
                feat[f"zscore_{k}"] = (p - roll_mean) / (roll_std + eps)
                upper_band = roll_mean + (2.0 * roll_std)
                lower_band = roll_mean - (2.0 * roll_std)
                feat[f"bollinger_pctB_{k}"] = (p - lower_band) / ((upper_band - lower_band) + eps)

        delta = p.diff()
        gain = delta.where(delta > 0, 0.0).shift(1).rolling(14).mean()
        loss = -delta.where(delta < 0, 0.0).shift(1).rolling(14).mean()
        rs = gain / (loss + eps)
        feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        return feat


class FractionalMemoryStrategy(BaseFeatureStrategy):
    def __init__(self, optimal_d: float, threshold: float = 1e-4, max_window: int = 100):
        self.d = optimal_d
        self.threshold = threshold
        self.max_window = max_window
        self.weights_forward = self._compute_weights(max_window)
        self.actual_max_len = len(self.weights_forward)

    def _compute_weights(self, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w)

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]

        def apply_frac_diff(x):
            x_rev = x[::-1]
            valid_len = min(len(x_rev), self.actual_max_len)
            return np.dot(self.weights_forward[:valid_len], x_rev[:valid_len])

        feat[f"frac_diff_d{self.d:.2f}"] = p.rolling(self.max_window, min_periods=10).apply(apply_frac_diff, raw=True)
        return feat


class VolatilityDynamicsStrategy(BaseFeatureStrategy):
    def __init__(self, dynamic_lags: list[int], has_asymmetric_vol: bool):
        self.lags = dynamic_lags
        self.has_asymmetric = has_asymmetric_vol

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        r = df["log_return"]

        feat["vol_parkinson"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(h / l) ** 2))
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        gk_core = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
        feat["vol_garman_klass"] = np.sqrt(np.maximum(0.0, gk_core))
        rs_core = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        feat["vol_rogers_satchell"] = np.sqrt(np.maximum(0.0, rs_core))

        window = 20
        o_prev_c = np.log(o / c.shift(1))
        c_prev_o = np.log(c / o)
        var_o = o_prev_c.rolling(window).var()
        var_c = c_prev_o.rolling(window).var()
        var_rs = pd.Series(rs_core, index=df.index).rolling(window).mean()
        k_val = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        feat["vol_yang_zhang"] = np.sqrt(np.maximum(0.0, var_o + k_val * var_c + (1.0 - k_val) * var_rs))

        if self.has_asymmetric:
            downside_ret = r.where(r < 0, 0.0)
            feat["vol_downside_dev_20"] = downside_ret.rolling(20).std(ddof=1)
            feat["vol_upside_dev_20"] = r.where(r > 0, 0.0).rolling(20).std(ddof=1)
            feat["vol_semi_variance_ratio_raw"] = feat["vol_downside_dev_20"] / (feat["vol_upside_dev_20"] + eps)

        feat["skewness_20"] = r.rolling(20).skew()
        feat["kurtosis_20"] = r.rolling(20).kurt()
        return feat


class VolatilityJumpDiffusionStrategy(BaseFeatureStrategy):
    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0)
        rv = (r**2).rolling(self.window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(self.window).sum()

        feat["jump_diffusion_component"] = np.maximum(rv - bv, 0.0) / (rv + eps)
        feat["continuous_vol_ratio"] = bv / (rv + eps)
        feat["jump_signed_shock"] = feat["jump_diffusion_component"] * np.sign(r)
        return feat


class MultiScaleComplexityStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0)

        def calc_pe(x):
            if len(x) < 6:
                return np.nan
            m, tau = 3, 1
            n = len(x) - (m - 1) * tau
            patterns = np.array([x[i : i + m * tau : tau] for i in range(n)])
            ranks = np.argsort(patterns, axis=1)
            _, counts = np.unique(ranks, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(6.0)

        feat["chaos_permutation_entropy_20"] = r.rolling(20).apply(calc_pe, raw=True)
        binary_seq = (r > 0).astype(int)
        feat["lz_complexity_proxy_20"] = (binary_seq.diff().abs()).rolling(20).mean()
        return feat


class WaveletMultiResolutionStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        p_s1 = p.shift(1)
        feat["wavelet_detail_d1"] = (p - p_s1) / np.sqrt(2.0)
        feat["wavelet_detail_d2"] = ((p + p_s1) - (p.shift(2) + p.shift(3))) / 2.0
        feat["wavelet_approx_a3"] = p.rolling(8).mean()
        feat["wavelet_energy_ratio"] = (feat["wavelet_detail_d1"] ** 2) / ((feat["wavelet_detail_d2"] ** 2) + eps)
        return feat


class IntradayShadowPressureStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]
        candle_range = (h - l) + eps
        body = (c - o).abs()
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        feat["shadow_asymmetry_ratio"] = (upper_shadow - lower_shadow) / candle_range
        feat["buying_tail_power"] = (lower_shadow / candle_range) * v
        feat["selling_tail_power"] = (upper_shadow / candle_range) * v
        feat["body_efficiency_ratio"] = body / candle_range
        return feat


class KinematicDynamicsStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        h, l = df["high_adj"], df["low_adj"]
        v = df["volume"]
        r = df["log_return"].fillna(0)

        feat["kinematic_velocity"] = r
        feat["kinematic_acceleration"] = r - r.shift(1)
        feat["kinematic_jerk"] = feat["kinematic_acceleration"] - feat["kinematic_acceleration"].shift(1)

        roll_std = p.shift(1).rolling(20).std(ddof=1)
        bb_width = 4.0 * roll_std
        p_prev = p.shift(1)
        tr1 = h - l
        tr2 = (h - p_prev).abs()
        tr3 = (l - p_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.shift(1).rolling(20).mean()
        kc_width = 3.0 * atr

        feat["kinematic_squeeze_ratio"] = bb_width / (kc_width + eps)
        clv = ((p - l) - (h - p)) / (h - l + eps)
        feat["kinematic_clv_vol"] = clv * v
        feat["kinematic_clv_vol_roll20"] = feat["kinematic_clv_vol"].rolling(20).mean()
        return feat


class GMMRegimeStrategy(BaseFeatureStrategy):
    def __init__(self, window: int = 120, update_freq: int = 20):
        self.window = window
        self.update_freq = update_freq

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0)
        vol = r.rolling(20).std(ddof=1).fillna(0)
        n = len(df)
        gmm_bull = np.zeros(n)
        gmm_bear = np.zeros(n)
        X = np.column_stack((r.values, vol.values))

        for i in range(self.window, n, self.update_freq):
            start_idx = max(0, i - self.window)
            X_train = np.nan_to_num(X[start_idx:i])
            pred_end = min(n, i + self.update_freq)
            X_test = np.nan_to_num(X[i:pred_end])
            try:
                if np.var(X_train[:, 0]) > 1e-8:
                    gmm = GaussianMixture(n_components=2, random_state=42, n_init=1)
                    gmm.fit(X_train)
                    probs = gmm.predict_proba(X_test)
                    bull_idx = np.argmax(gmm.means_[:, 0])
                    bear_idx = 1 - bull_idx
                    gmm_bull[i:pred_end] = probs[:, bull_idx]
                    gmm_bear[i:pred_end] = probs[:, bear_idx]
            except Exception:
                pass

        feat["gmm_prob_bull"] = gmm_bull
        feat["gmm_prob_bear"] = gmm_bear
        feat.iloc[: self.window, :] = np.nan
        return feat


class VolumePriceDivergenceStrategy(BaseFeatureStrategy):
    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df["high_adj"], df["low_adj"], df["close_adj"]
        v = df["volume"]
        r = df["log_return"].fillna(0)

        clv = ((c - l) - (h - c)) / ((h - l) + eps)
        vol_money_flow = clv * v
        feat["cmf_20"] = vol_money_flow.rolling(self.window).sum() / (v.rolling(self.window).sum() + eps)
        feat["cmf_slope_5"] = feat["cmf_20"] - feat["cmf_20"].shift(5)

        obv_direction = np.sign(r)
        obv = (obv_direction * v).cumsum()
        obv_ma = obv.rolling(self.window).mean()
        obv_std = obv.rolling(self.window).std(ddof=1) + eps
        feat["obv_zscore_20"] = (obv - obv_ma) / obv_std
        feat["obv_slope_5"] = (obv - obv.shift(5)) / (v.rolling(self.window).mean() + eps)

        vpt = (r * v).cumsum()
        vpt_ma = vpt.rolling(self.window).mean()
        vpt_std = vpt.rolling(self.window).std(ddof=1) + eps
        feat["vpt_zscore_20"] = (vpt - vpt_ma) / vpt_std

        norm_price_ret = (c / c.shift(self.window) - 1.0)
        feat["cmf_price_divergence"] = feat["cmf_20"] - norm_price_ret
        return feat


class DirectionalPressureStrategy(BaseFeatureStrategy):
    def __init__(self, window: int = 14):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]
        candle_range = (h - l) + eps
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        body = c - o

        buy_strength = (c - l) / candle_range
        sell_strength = (h - c) / candle_range
        feat["net_pressure_ratio"] = buy_strength - sell_strength
        feat["net_pressure_volume"] = feat["net_pressure_ratio"] * np.log(v + 1.0)
        feat["net_pressure_vol_roll"] = feat["net_pressure_volume"].rolling(self.window).mean()

        feat["wick_absorption_direction"] = (lower_wick - upper_wick) / candle_range
        feat["wick_absorption_thrust"] = feat["wick_absorption_direction"] * (v / (v.rolling(self.window).mean() + eps))

        body_ratio = body / candle_range
        feat["body_direction_momentum"] = body_ratio.rolling(5).mean()
        feat["body_direction_accel"] = feat["body_direction_momentum"] - feat["body_direction_momentum"].shift(2)
        return feat


class SignedVolatilityDivergenceStrategy(BaseFeatureStrategy):
    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0)
        c, o = df["close_adj"], df["open_adj"]

        downside_ret = r.where(r < 0, 0.0)
        upside_ret = r.where(r > 0, 0.0)
        downside_var = (downside_ret**2).rolling(self.window).mean()
        upside_var = (upside_ret**2).rolling(self.window).mean()

        feat["vol_directional_bias"] = (upside_var - downside_var) / (upside_var + downside_var + eps)
        feat["vol_directional_accel"] = feat["vol_directional_bias"] - feat["vol_directional_bias"].shift(3)
        daily_range = np.log(df["high_adj"] / df["low_adj"])
        feat["signed_expansion_thrust"] = np.sign(c - o) * daily_range
        return feat


class MarketContextStrategy(BaseFeatureStrategy):
    def __init__(self, window_short: int = 20, window_long: int = 60):
        self.ws = window_short
        self.wl = window_long

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if "mkt_return" not in df.columns:
            return feat

        r_asset = df["log_return"].fillna(0)
        r_mkt = df["mkt_return"].fillna(0)

        feat["mkt_relative_return"] = r_asset - r_mkt
        feat["mkt_rs_cum_20"] = r_asset.rolling(self.ws).sum() - r_mkt.rolling(self.ws).sum()
        feat["mkt_rs_momentum"] = feat["mkt_rs_cum_20"] - feat["mkt_rs_cum_20"].shift(5)

        cov_60 = r_asset.rolling(self.wl).cov(r_mkt)
        var_mkt_60 = r_mkt.rolling(self.wl).var()
        rolling_beta = cov_60 / (var_mkt_60 + eps)
        feat["mkt_rolling_beta_60"] = rolling_beta.clip(lower=-1.0, upper=3.5)
        feat["mkt_beta_shock"] = feat["mkt_rolling_beta_60"] - feat["mkt_rolling_beta_60"].shift(5)

        vol_asset_20 = r_asset.rolling(self.ws).std(ddof=1)
        vol_mkt_20 = r_mkt.rolling(self.ws).std(ddof=1)
        feat["mkt_vol_ratio_20"] = vol_asset_20 / (vol_mkt_20 + eps)
        feat["mkt_correlation_20"] = r_asset.rolling(self.ws).corr(r_mkt).fillna(0)
        feat["mkt_divergence_signed"] = np.sign(r_asset) * np.sign(r_mkt) * (r_asset - r_mkt)
        return feat


class LiquidityMicrostructureStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df["high"], df["low"], df["close"]
        v = df["volume"]
        r = df["log_return"]

        dollar_volume = c * v
        amihud_raw = r.abs() / dollar_volume.replace(0, np.nan)
        feat["liq_amihud_raw"] = amihud_raw
        feat["liq_amihud_z"] = (amihud_raw - amihud_raw.rolling(20).mean()) / (amihud_raw.rolling(20).std() + eps)

        def calc_autocorr_lag1(x):
            return pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0

        feat["liq_roll_measure_20"] = r.rolling(20).apply(calc_autocorr_lag1, raw=False)
        feat["liq_log_turnover"] = np.log(dollar_volume + eps)
        feat["liq_turnover_ratio_20"] = dollar_volume / (dollar_volume.shift(1).rolling(20).mean() + eps)
        return feat


class OrderFlowToxicityStrategy(BaseFeatureStrategy):
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if all(col in df.columns for col in ["bid_size_1", "ask_size_1"]):
            order_imbalance = df["bid_size_1"] - df["ask_size_1"]
        elif all(col in df.columns for col in ["bid_size", "ask_size"]):
            order_imbalance = df["bid_size"] - df["ask_size"]
        else:
            c, h, l = df["close_adj"], df["high_adj"], df["low_adj"]
            v = df["volume"]
            buy_pressure = (c - l) / (h - l + eps)
            sell_pressure = (h - c) / (h - l + eps)
            order_imbalance = (buy_pressure - sell_pressure) * v

        feat["flow_imbalance_proxy"] = order_imbalance
        feat["flow_imbalance_zscore"] = (order_imbalance - order_imbalance.rolling(20).mean()) / (
            order_imbalance.rolling(20).std() + eps
        )
        v_series = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
        vol_bucket = v_series.rolling(10).sum()
        imbalance_bucket = order_imbalance.abs().rolling(10).sum()
        feat["flow_vpin_10"] = imbalance_bucket / (vol_bucket + eps)
        return feat


class Layer10FeatureRouter:
    def __init__(self, payload: dict):
        self.payload = payload
        self.registry: dict[str, BaseFeatureStrategy] = {}
        self._dispatch()

    def _dispatch(self):
        flags = self.payload.get("flags", {})
        lags = self.payload.get("dynamic_lags", [1, 3, 5, 10, 20])
        opt_d = self.payload.get("optimal_d", 1.0)

        self.registry["Baseline & Geometry"] = BaselineStrategy(dynamic_lags=lags)
        self.registry["Liquidity Microstructure"] = LiquidityMicrostructureStrategy()
        self.registry["Kinematics & Flow"] = KinematicDynamicsStrategy()
        self.registry["GMM State Clustering"] = GMMRegimeStrategy(window=120, update_freq=20)
        self.registry["Intraday Shadow Pressure"] = IntradayShadowPressureStrategy()
        self.registry["Wavelet Multi-Resolution"] = WaveletMultiResolutionStrategy()
        self.registry["Volume-Price Flow Divergence"] = VolumePriceDivergenceStrategy(window=20)
        self.registry["Directional Candle Pressure"] = DirectionalPressureStrategy(window=14)
        self.registry["Signed Volatility Bias"] = SignedVolatilityDivergenceStrategy(window=20)
        self.registry["Market Exogenous Context (VN-INDEX)"] = MarketContextStrategy(window_short=20, window_long=60)

        if flags.get("has_long_trend", True):
            self.registry["Trend & Momentum"] = TrendMomentumStrategy(dynamic_lags=lags)
        if flags.get("has_short_reversion", True):
            self.registry["Oscillators & Reversion"] = OscillatorMeanReversionStrategy(dynamic_lags=lags)
        if 0.0 < opt_d < 1.0:
            self.registry["Fractional Memory"] = FractionalMemoryStrategy(optimal_d=opt_d)
        if flags.get("has_vol_clustering", True):
            self.registry["Volatility Dynamics"] = VolatilityDynamicsStrategy(
                dynamic_lags=lags,
                has_asymmetric_vol=flags.get("has_asymmetric_vol", False),
            )
        if flags.get("has_vol_jumps", False):
            self.registry["Volatility Jump Diffusion"] = VolatilityJumpDiffusionStrategy(window=20)
        if flags.get("is_high_complexity", False):
            self.registry["Multi-Scale Complexity"] = MultiScaleComplexityStrategy()
        if flags.get("is_volume_significant", True):
            self.registry["Order Flow Toxicity (Proxy)"] = OrderFlowToxicityStrategy()

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frames = [strategy.construct(df) for strategy in self.registry.values()]
        return pd.concat(feature_frames, axis=1)


class Stage2DiagnosticEngine:
    def fit(self, df_train: pd.DataFrame) -> dict:
        return Stage2SingleAssetDGPScanner.execute(df_train)


class Stage3FeatureEngine:
    def __init__(self, payload: dict):
        self.router = Layer10FeatureRouter(payload)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.router.execute(df)