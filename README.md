# Overcoming Microstructure Frictions and Concept Drift via Calibrated Meta-Labeling: An Econometrically Grounded Architecture for Emerging Equity Markets

![Python 3.10+](https://www.python.org/)
![License: MIT](https://opensource.org/licenses/MIT)
![Research Phase-success.svg)](#)

This repository constitutes **Phase 3** of the empirical quantitative research continuum evaluating predictability and execution architectures in emerging financial markets (HOSE, Vietnam).

- **Phase 1 Repository:** Quant-ML-Predictability-Audit
- **Phase 2 Repositories:** Empirical-Evaluation-of-Asymmetric-Thresholding (Version 1) & Part-2 (Version 2)

---

## 1. Executive Summary & Problem Formulation

In Phase 2, a ternary Extreme Gradient Boosting (XGBoost) model predicting discrete directional barriers $y \in \{-1, 0, 1\}$ was subjected to static out-of-fold decision boundaries. While reporting strong validation accuracy (43%–69%), live transactional simulation across realistic market conditions (15 bps base duties + 5 bps dynamic slippage) triggered catastrophic drawdowns: **-42.59% Net Return**, **-73.97% Maximum Drawdown (MDD)**, and a **-0.45 Sharpe ratio**.

Phase 2 identified two structural failure modes:

1. **Position Entrapment via Concept Drift:** When market regimes transitioned from momentum to mean-reverting phases, static thresholds became obsolete, holding equity exposure through systematic downturns.

2. **Probability Margin Distortion & Microstructure Chattering:** Raw tree outputs exhibited extreme margin polarization, generating micro-turnover that eroded theoretical returns into execution fees (turnover exceeding 400x). **Phase 3 resolves these failures** by introducing a closed-loop **Two-Stage Meta-Labeling Architecture** combined with **Out-of-Fold (OOF) Isotonic Probability Calibration**, an **Execution Deadband Buffer (15%)**, and synchronization with Vietnam's **$T+2.5$ settlement constraint**.

---

## 2. Quantitative Architecture

```
+-----------------------------------------------------------------------------------+
| STAGE 1: PHYSICAL INTEGRITY & FORWARD ADJUSTMENT                                  |
| Raw OHLCV + VN-Index -> Forward-Adjustment Protocol -> Geometric Bar Auditing     |
+-----------------------------------------------------------------------------------+
|
+-----------------------------------------------------------------------------------+
| STAGES 2 & 3: ECONOMETRIC FEATURE GENERATION ENGINE                               |
| Dynamic Sensor Payloads: Wavelets, Barndorff-Nielsen Jumps, Kinematic Lag Vector  |
+-----------------------------------------------------------------------------------+
|
+-----------------------------------------------------------------------------------+
| STAGE 4: MULTI-PHASE STATISTICAL PRUNING                                          |
| Linear VIF (< 6.0) -> HRP Spearman Clustering -> Granger F-Test -> Conditional MI |
+-----------------------------------------------------------------------------------+
|
+-----------------------------------------------------------------------------------+
| STAGE 5: TWO-STAGE META-LABELING PIPELINE                                         |
| * Primary Filter: S_t = I(Close > EMA50) x I(r_mkt_20 > -0.006) (Macro Gate)      |
| * Secondary Classifier: P(Take-Profit Hit | S_t = 1) via XGBoost + Purged CV      |
+-----------------------------------------------------------------------------------+
|
+-----------------------------------------------------------------------------------+
| STAGE 6: REALISTIC EXECUTION & MICROSTRUCTURE SIMULATOR                           |
| OOF Isotonic Calibration -> Confidence Sizing -> 15% Deadband -> T+2.5 Min-Hold   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Core Out-of-Sample Empirical Results

Evaluated on **1,655 out-of-sample (OOS) daily trading bars** (June 2018 – June 2026) for DIG equity (HOSE) across 5 Purged and Embargoed Time-Series Walk-Forward folds:

| Performance Metric (Net of 20 bps Friction) | Production Meta-Model | Buy & Hold Benchmark | Empirical Improvement |
| :--- | :---: | :---: | :---: |
| **Cumulative Net Return** | **+89.65%** | +66.87% | **+22.78% Alpha** |
| **Annualized Return (CAGR)** | **10.24%** | 8.11% | **+2.13%** |
| **Annualized Volatility** | **15.35%** | ~42.50% | **-27.15% (Risk Suppression)** |
| **Net Sharpe Ratio ($R_f=0$)** | **0.71** | 0.41 | **+0.30** |
| **Sortino Ratio (Downside Adjusted)** | **0.63** | ~0.35 | **+0.28** |
| **Calmar Ratio (CAGR / MDD)** | **0.40** | 0.09 | **+0.31** |
| **Maximum Drawdown (MDD)** | **-25.82%** | -88.50% | **+62.68% Preservation** |
| **Market Exposure** | **39.82%** | 100.00% | Defends Capital in Cash |
| **Cumulative Portfolio Turnover** | **81.41x** | 1.00x | Chattering Suppressed |
| **Total Cumulative Transaction Friction** | **16.28%** | ~0.30% | Accounted For in Net Return |

---

## 4. Methodological Comparison Across Continuum Phases

| Evaluation Criterion | Phase 1 (Feature Audit) | Phase 2 (Static Thresholding) | Phase 3 (Calibrated Meta-Labeling) |
| :--- | :---: | :---: | :---: |
| **Target Space** | Ternary $y \in \{-1, 0, 1\}$ | Ternary $y \in \{-1, 0, 1\}$ | Decoupled: Trend Filter + Binary Meta $Y \in \{0, 1\}$ |
| **Probability Calibration** | None (Raw Softmax) | None (Raw Tree Margin) | **Out-of-Fold Non-parametric Isotonic Regression** |
| **Sizing Function** | Unit Discrete $\{-1, 0, 1\}$ | Static Volatility Targeting | **Continuous Convex Sizing $[w_{base}=0.25 \to 1.0]$** |
| **Execution Buffer** | Unbuffered $T+1$ | Unbuffered $T+1$ | **15% Rebalance Deadband + 3-Day Holding Minimum** |
| **Net Return** | -15.82% | -42.59% | **+89.65%** |
| **Net Sharpe Ratio** | -0.64 | -0.45 | **0.71** |
| **Maximum Drawdown** | -73.04% | -73.97% | **-25.82% (vs. -88.50% Benchmark)** |
| **System Status** | Methodological Proof | Microstructure Failure | **Production-Ready, Fully Rehabilitated** |

---

## 5. Key Econometric Insights

1. **Multi-Resolution Wavelet Dominance (27.15% Relative Gain):**

Haar Wavelet level-3 low-frequency approximation (`wavelet_approx_a3_zscaled_lag1`) is the single most predictive feature, filtering out market microstructure noise to detect institutional accumulation.

2. **Fat-Tail & Skewness Governors (>45% Aggregate Gain):**

Higher-order statistical moments (`kurtosis_20_zscaled_lag2`, `kurtosis_20_momentum`, and `skewness_20_momentum`) serve as quantitative early warnings, de-allocating exposure before volatility expansion.

3. **Liquidity Friction Defense:**

The combination of the 15% deadband filter and the $T+2.5$ minimum holding period compressed turnover by over **80%** relative to unconstrained models, keeping friction costs down to 16.28% over 6.5 years.

---

## 6. Installation & Quickstart

```
# Clone the repository
git clone https://github.com/LamTong21/Empirical-Evaluation-of-Asymmetric-Thresholding-Part-3.git
cd Empirical-Evaluation-of-Asymmetric-Thresholding-Part-3

# Install dependencies
pip install -r requirements.txt

# Execute pipeline simulation
python -m src.simulator 
```

---

### Hướng dẫn Tổ chức Mã nguồn (Clean Code Modularization)

Để chuyển notebook thành mã nguồn sản phẩm đạt chuẩn kiểm thử:

- Đưa `Layer1DataIntegrity`, `Layer2MicrostructureTopologies` và `Layer0TripleBarrier` vào `src/data_integrity.py` và `src/labeling.py`[cite: 1].
- Đưa các lớp Strategy (`WaveletMultiResolutionStrategy`, `VolatilityJumpDiffusionStrategy`, `KinematicDynamicsStrategy`, v.v.) và `Layer10FeatureRouter` vào `src/econometric_engine.py`[cite: 1].
- Đưa `Layer12RedundancyControl`, `Layer13PredictiveDiagnostics` và `Layer14LagTransformEngine` vào `src/feature_selection.py`[cite: 1].
- Đưa logic huấn luyện Optuna, XGBoost objective và Isotonic calibration vào `src/meta_engine.py` và `src/calibration.py`[cite: 1].
- Đưa lớp mô phỏng `LowTurnoverMetaSimulator` vào `src/simulator.py`[cite: 1].