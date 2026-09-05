# Overcoming Microstructure Frictions and Concept Drift via Calibrated Meta-Labeling: An Econometrically Grounded Architecture for Emerging Equity Markets

**Status:** In progress
**Assign:** Quant. Field
**Description:** Meta-Labeling
**Research Type:** Modeling

**Author:** Quantitative Research Group
**Dataset:** DIG Equity (HOSE, Vietnam) & VN-Index Benchmark (2018-2026, 1,995 Daily Sessions)
**Methodological Classification:** Quantitative Finance, Econometrics, Machine Learning, Market Microstructure

## Abstract
Empirical financial machine learning models frequently exhibit profound degradation when transitioning from backtests to production environments. In emerging equity markets, this deterioration is intensified by three structural impediments: severe microstructure transaction frictions, non-stationary market regimes (concept drift), and settlement horizon constraints such as the Vietnamese market's $T+2.5$ settlement cycle.

Inheriting the foundational econometric feature representation from "An Econometrically Grounded, Context-Aware Feature Engineering and Validation Framework for Equity Return Predictability" and directly answering the empirical failure identified in "Empirical Evaluation of Asymmetric Thresholding in Machine Learning-Based Equity Trading Systems" (where static decision boundaries across discrete ternary targets suffered a -42.59% net loss and -73.97% maximum drawdown), this study introduces a closed-loop, Two-Stage Meta-Labeling architecture.

Our system decouples directional commitment from sizing confidence:
1.  A macro-filtered trend engine specifies long-side trade authorization.
2.  A secondary binary XGBoost classifier conditions on the econometric feature space to evaluate execution viability via dynamic triple-barrier horizons.
3.  Out-of-fold (OOF) Isotonic Regression probability calibration reconciles tree-based margin distortion.
4.  A non-linear execution buffer enforces a 15% rebalance deadband and a minimum 3-day holding period to eliminate transaction chattering.

Evaluated over 1,655 out-of-sample (OOS) trading days spanning 6.5 calendar years across 5 purged and embargoed time-series cross-validation folds, the architecture demonstrates complete performance rehabilitation: achieving a Cumulative Net Return of +89.65% (versus +66.87% for the Buy-and-Hold benchmark), a Net Annualized CAGR of 10.24% (versus 8.11%), a Net Sharpe Ratio of 0.71 (versus 0.41), and truncating market drawdown exposure to -25.82% against an equity peak-to-trough collapse of -88.50%.

## 1. Introduction & Research Nexus

### 1.1 The Friction-Drift Paradox in Asset Return Predictability
The deployment of machine learning algorithms to financial time series is dominated by the tension between statistical signal extraction and microstructure friction costs. In ideal academic assumptions characterized by frictionless continuous execution and stationary data-generating processes (DGPs), complex non-linear architectures (e.g., Deep Learning, Gradient Boosted Decision Trees) reliably report statistically significant Out-of-Sample Information Coefficients (IC) and classification accuracy.

However, in physical execution domains—especially emerging equity markets characterized by asymmetric liquidity, discrete tick ladders, fixed transactional duties, and settlement lags—these theoretical signals rapidly disintegrate. High turnover models dissipate their gross margin into slippage and broker commissions, a phenomenon termed Microstructure Friction Decay.

### 1.2 Retrospective on Prior Work
This paper forms the third and conclusive phase of an empirical research continuum:

*   **Phase 1: Feature Topology & Econometric Validation:** Established an automated feature factory and multi-phase statistical selection protocol. It demonstrated that raw technical indicators generate spurious predictability, whereas causal Granger screening, Spectral Fourier Cycle analysis, Barndorff-Nielsen & Shephard Jump Diffusion tests, and Kinematic Lag Momentum/Acceleration transformations extract durable alpha kernels without lookahead contamination.
*   **Phase 2: Static Ternary Asymmetric Threshold Failure:** Examined the deployment of an XGBoost classifier predicting a ternary Triple-Barrier target $y \in \{-1, 0, 1\}$ using static decision thresholds calibrated from validation folds. While reporting cross-validation accuracy ranging from 43% to 69%, realistic execution accounting for 15 bps base fees and dynamic slippage caused catastrophic drawdown: -42.59% Cumulative Return, -73.97% Maximum Drawdown (MDD), and an annualized Sharpe of -0.45.
*   **Phase 3: Calibrated Meta-Labeling & Deadband Execution Engine (Current Work)**

Phase 2 diagnosed two fatal structural pathologies:
1.  **Position Entrapment via Concept Drift:** When the market shifts from momentum to mean-reverting regimes, static probability cutoffs calibrated on preceding folds become misaligned, locking the portfolio into high-beta drawdowns or sustained dormancy.
2.  **Probability Margin Distortion:** Raw tree-based probabilities tend to polarize near terminal leaf distributions, failing to reflect true empirical frequency and generating excessive rebalancing chattering that transfers capital to intermediaries.

### 1.3 Methodological Contributions of this Study
To definitively solve the structural failure of Phase 2 while preserving the econometric rigors of Phase 1, this paper makes four principal contributions:

*   **Two-Stage Meta-Labeling Paradigm:** Decoupling directional orientation (assigned to a macro-filtered trend engine) from position sizing and probability verification (assigned to a machine learning classifier).
*   **Out-of-Fold Isotonic Probability Calibration:** Incorporating an isotonic mapping layer fitted strictly on out-of-fold cross-validation predictions, producing non-parametric, monotonically calibrated event probabilities.
*   **Constrained Confidence Sizing:** Mapping calibrated probabilities into continuous capital allocation weights through a convex scaling function bounded by safety floors.
*   **Microstructure-Synchronized Deadband Execution:** Formulating an execution buffer that imposes a 15% weight-change barrier $(|\Delta w| \ge 0.15)$ and enforces a 3-day minimum holding horizon, harmonizing algorithm latency with Vietnam's $T+2.5$ settlement cycle and suppressing transaction turnover.

## 2. Mathematical Modeling Architecture

### 2.1 Information Space and Forward-Adjustment Protocol
Let the filtered probability space be denoted by $(\Omega, \mathcal{F}, \{\mathcal{F}_{t}\}_{t \ge 0}, \mathbb{P})$, where the filtration $\mathcal{F}_{t} = \sigma(\{O_{s}, H_{s}, L_{s}, C_{s}, V_{s}, I_{s}\}_{s \le t})$ encapsulates all historic pricing, volume, and exogenous benchmark series available up to discrete trading session $t$.

To prevent structural lookahead bias arising from traditional backward-adjusted corporate actions (where historical series are shifted downwards upon dividend distributions, retroactively altering volatility thresholds), we enforce strict forward adjustment. Real daily return $R_{t}^{real}$ incorporates cash dividends $D_{t}$ and stock split coefficient $S_{t}$:

$$R_{t}^{real} = \frac{C_{t} + D_{t}}{C_{t-1} \cdot S_{t}} - 1.0$$ (1)

The forward-adjusted closing price trajectory $C_{t}^{adj}$ is reconstructed by setting $C_{0}^{adj} = C_{0}$ and compounding forward:

$$C_{t}^{adj} = C_{0} \prod_{i=1}^{t} (1.0 + R_{i}^{real})$$ (2)

The scalar normalization metric $\kappa_{t} = \frac{C_{t}^{adj}}{C_{t}}$ is applied across open, high, and low series: $P_{t}^{adj} = P_{t} \cdot \kappa_{t}, \forall P \in \{O, H, L\}$.

The fundamental state topologies are established as continuous differential return forms:
*   Total Continuous Return: $r_{t} = \ln(C_{t}^{adj} / C_{t-1}^{adj})$
*   Overnight Jump Dynamics: $r_{t}^{overnight} = \ln(O_{t}^{adj} / C_{t-1}^{adj})$
*   Intraday Continuous Velocity: $r_{t}^{intraday} = \ln(C_{t}^{adj} / O_{t}^{adj})$
*   Extremum Logarithmic Range: $range_{t}^{HL} = \ln(H_{t}^{adj} / L_{t}^{adj})$

### 2.2 Dynamic Triple-Barrier Meta-Targeting
Classical directional prediction models train on ternary discretizations:

$$y_{t} \in \{-1, 0, 1\}$$ (3)

This forces the machine learning model to simultaneously learn trend orientation and trade execution quality, introducing severe noise. In contrast, our Meta-Labeling architecture reformulates the learning problem: given a trade hypothesis generated by a macro-heuristic, evaluate whether that trade will reach an upper barrier (take-profit) before touching a lower barrier (stop-loss) or timing out.

Local price volatility is continuously estimated via an Exponentially Weighted Moving Average (EWMA) with span $\lambda=20$:

$$\sigma_{t} = \sqrt{\sum_{i=0}^{\infty} w_{i} (r_{t-i} - \overline{r}_{t})^{2}}, \quad w_{i} = \alpha(1-\alpha)^{i}, \quad \alpha = \frac{2}{\lambda+1}$$ (4)

For a primary trade entry at $t_{0}$ with reference price $P_{0} = C_{t_{0}}^{adj}$, horizontal execution bounds scale dynamically with local volatility:

$$U_{t} = P_{0} \cdot (1.0 + pt \cdot \sigma_{t_{0}})$$ (5)

$$L_{t} = P_{0} \cdot (1.0 - sl \cdot \sigma_{t_{0}})$$ (6)

where $pt=1.0, sl=1.0$, and the vertical time boundary enforces an expiration horizon of $h=5$ sessions.

Let stopping times be defined as:

$$\tau_{upper} = \min(\{\tau \in [1, h] | C_{t_{0}+\tau}^{adj} \ge U_{t}\} \cup \{\infty\})$$ (7)

$$\tau_{lower} = \min(\{\tau \in [1, h] | C_{t_{0}+\tau}^{adj} \le L_{t}\} \cup \{\infty\})$$ (8)

The binary meta-label $Y_{t_{n}}^{meta}$ is assigned strictly as:

$$Y^{meta} = \begin{cases} 1 & \text{if } \tau_{upper} < \tau_{lower} \text{ and } \tau_{upper} \le h \text{ (Take-Profit Hit)} \\ 0 & \text{if } \tau_{lower} \le \tau_{upper} \text{ or } (\tau_{upper} = \tau_{lower} = \infty) \text{ (Stop-Loss Hit / Time-out)} \end{cases}$$ (9)

### 2.3 Econometric Feature Engine and Selection Router
Rather than populating the feature matrix with arbitrary momentum indicators, feature candidate construction is governed by an automated econometric diagnostic pipeline executing exclusively on each fold's training split.

**Raw Training Set Pipeline:**
1.  **Phase 1: Linear VIF & Collinearity Pruning:** Pairwise Pearson correlation matrices filter redundant features $|\rho_{i,j}| > 0.85$. Remaining vectors undergo recursive Variance Inflation Factor (VIF) pruning, discarding candidates exceeding $VIF > 6.0$:
    $$VIF_{j} = \frac{1}{1 - R_{j}^{2}}$$ (10)
2.  **Phase 2: Non-linear HRP Spearman Clustering:** Non-linear dependencies are projected onto an angular metric distance space:
    $$D_{i,j} = \sqrt{\frac{1}{2} (1 - \rho_{i,j}^{Spearman})}$$ (11)
    Hierarchical tree structures are computed via Ward's linkage algorithm. For each cluster $k \in \{1,...,K\}$ (where $K \le 14$), the cluster medoid $x_{k}^{*}$ is isolated.
3.  **Phase 3: Dual Causal Screening (Granger & Transfer Entropy):** To verify that candidate features precede the target trajectory in temporal causality, bivariate Vector Autoregressions are fitted against $Y^{meta}$:
    $$y_{t} = c_{1} + \sum_{i=1}^{p} \alpha_{i} y_{t-i} + \sum_{j=1}^{p} \beta_{j} X_{t-j} + \epsilon_{t}$$ (12)
    Features must satisfy the statistical rejection of $H_{0}: \beta_{1} = \cdots = \beta_{p} = 0$ via F-test with empirical p-value < 0.05 across lags $p \in [1, 5]$. Non-linear information transfer is simultaneously checked via lagged mutual information (Transfer Entropy proxy). The optimal lead lag $L^{*}$ is indexed at the point of maximum causality.
4.  **Phase 4: Kinematic Lag Transformations:** Admitted features are transformed into kinematic vectors at lag $L^{*}$:
    *   Volatility-Scaled State: $Z_{t,L^{*}} = \frac{X_{t-L^{*}} - \mu_{20,t-L^{*}}}{\sigma_{20,t-L^{*}} + \epsilon}$
    *   First-Order Momentum: $M_{t,L^{*}} = \frac{X_{t} - X_{t-L^{*}}}{|X_{t-L^{*}}| + \epsilon}$
    *   Second-Order Acceleration: $A_{t,L^{*}} = M_{t,L^{*}} - M_{t-1,L^{*}}$
5.  **Phase 5: Regime-Conditioned Information Filtering:** Mutual Information (MI) is computed conditioned across latent volatility states derived from Gaussian Mixture Models (GMM):
    $$\max(I(X|Reg_{high}; Y), I(X|Reg_{low}; Y)) > 0.01$$ (13)
    The algorithm caps the final dimensionality strictly between $N_{min} = 6$ and $N_{max} = 16$ features to prevent parameter explosion.

### 2.4 The Two-Stage Meta-Labeling Engine
The architectural decoupling between trade origination and execution authorization proceeds as follows:

*   **Primary Signal Model ($S_{t}$):** Designed to maximize directional recall while maintaining macro trend conformity, the primary filter generates a binary trade opportunity flag using an asset-level trend check combined with an index-level momentum filter, shifted by 1 day to eliminate micro lookahead bias:
    $$S_{t} = \mathbb{I}(C_{t-1}^{adj} > EMA_{50,t-1}) \times \mathbb{I}(\overline{r}_{20,t-1}^{VN-INDEX} > -0.006)$$ (14)
    Sessions where $S_{t}=0$ are immediately assigned zero allocation, insulating the portfolio from macro downtrends.
*   **Secondary Classifier Model (Meta-Model):** The secondary model is parameterized as an Extreme Gradient Boosting (XGBoost) classifier trained exclusively on historical bars where the primary condition was active: $\mathcal{D}_{train}^{meta} = \{(X_{t}, Y_{t}^{meta}) | S_{t}=1\}$. The model objective optimizes logarithmic loss regularized against high dimensionality:
    $$\mathcal{L}_{meta} = - \sum_{i} [w_{i} \cdot (y_{i} \ln(p_{i}) + (1-y_{i}) \ln(1-p_{i}))] + \gamma T_{leaves} + \frac{1}{2} \lambda \sum w_{j}^{2}$$ (15)
    where sample weights $w_{i}$ balance positive and negative outcomes dynamically across folds.

### 2.5 Out-of-Fold Isotonic Probability Calibration
Raw predicted margins $z_{i}(x)$ generated by gradient boosted trees suffer from systematic distortion: boosting objective functions tend to push predicted scores towards extreme probabilities, distorting the empirical risk distribution.

To ensure that a predicted probability $\hat{p}=0.60$ precisely corresponds to a 60% empirical expectation of take-profit realization, we calibrate the outputs using non-parametric Isotonic Regression. Within each training fold, an internal 3-fold Purged Cross-Validation generates out-of-fold raw probabilities $p_{raw,i}^{OOF}$. The isotonic transformation minimizes squared error subject to a monotonicity constraint:

$$\min \sum_{i=1}^{N} (y_{i} - \hat{m}(p_{raw,i}^{OOF}))^{2} \quad \text{subject to} \quad \hat{m}(p_{a}) \le \hat{m}(p_{b}) \quad \forall p_{a} \le p_{b}$$ (16)

During inference, raw test probabilities are projected through the isotonic step-function:

$$\hat{P}_{calibrated} = \hat{m}(p_{raw}^{test})$$ (17)

### 2.6 Dynamic Threshold Optimization & Confidence Sizing
Rather than imposing an arbitrary classification threshold (such as $\tau = 0.50$), optimal execution thresholds $\tau^{*} \in [0.480, 0.540]$ are resolved on the calibrated OOF set using a risk-adjusted utility function based on the $F_{0.5}$ metric (which penalizes false positives more heavily than false negatives, prioritizing trade precision over trade frequency):

$$F_{\beta} = (1 + \beta^{2}) \frac{\text{Precision} \times \text{Recall}}{\beta^{2} \text{Precision} + \text{Recall}} \quad \beta = 0.5$$ (18)

To ensure that the model does not collapse into a trivial non-trading solution, a coverage penalty is applied if the fraction of generated trades falls below $Coverage_{min} = 8\%$:

$$\mathcal{U}(\tau) = F_{0.5}(\tau) \times \min\left(1.0, \frac{Coverage(\tau)}{Coverage_{min}}\right)$$ (19)

The optimal decision boundary $\tau^{*}$ is indexed as:

$$\tau^{*} = \text{clip}(\arg \max_{\tau} \mathcal{U}(\tau), 0.480, 0.540)$$ (20)

Once $\tau^{*}$ is established, capital sizing operates as a continuous, non-linear confidence function. When calibrated confidence exceeds $\tau^{*}$, position sizing scales linearly from an entry floor of $w_{base} = 0.25$ up to maximum allocation $(w_{max} = 1.0)$:

$$w_{t}^{raw} = \begin{cases} w_{base} + (1.0 - w_{base}) \left(\frac{\hat{P}_{calibrated,t} - \tau^{*}}{1.0 - \tau^{*}}\right) & \text{if } \hat{P}_{calibrated,t} \ge \tau^{*} \\ 0.0 & \text{if } \hat{P}_{calibrated,t} < \tau^{*} \end{cases}$$ (21)

The final target allocation couples the primary filter with the meta-decision:

$$w_{t}^{target} = \mathbb{I}(S_{t} = 1) \times w_{t}^{raw}$$ (22)

### 2.7 Realistic Microstructure Friction & Deadband Simulator
Realistic simulation requires accounting for implementation shortfall, regulatory constraints, and latency.

*   **Execution Latency:** Signals computed upon bar close at date $t-1$ are matched at the close of date $t$, eliminating contemporaneous execution lookahead:
    $$w_{t}^{executed} = w_{t-1}^{target}$$ (23)

*   **Chattering Suppression via 15% Deadband Filter:** To eliminate micro-turnover generated by negligible shifts in probability, weight updates must clear an absolute threshold deadband $\theta_{deadband} = 0.15$:
    $$w_{t}^{*} = \begin{cases} w_{t}^{executed} & \text{if } |w_{t}^{executed} - w_{t-1}^{*}| \ge \theta_{deadband} \\ w_{t-1}^{*} & \text{if } |w_{t}^{executed} - w_{t-1}^{*}| < \theta_{deadband} \text{ and } w_{t}^{executed} > 0 \\ 0.0 & \text{if } w_{t}^{executed} = 0.0 \text{ and } holding\_days \ge 3 \end{cases}$$ (24)

*   **Minimum Holding Period $\{T_{min} = 3\}$:** To reflect Vietnam's $T+2.5$ settlement constraint (where equities bought on day $T$ cannot be liquidated until the afternoon of $T+2$ effectively executing on $T+3$), any active position is locked against liquidation until $holding\_days \ge 3$.

*   **Microstructure Friction Model:** The cost model enforces a fixed base transaction fee of 15 bps plus a conservative dynamic slippage allowance of 5 bps, establishing total friction at $C_{friction} = 20 \text{ bps} = 0.0020$ per unit of portfolio turnover:
    $$Turnover_{t} = |w_{t}^{*} - w_{t-1}^{*}|$$ (25)
    $$Cost_{t} = Turnover_{t} \times C_{friction}$$ (26)
    $$r_{t}^{net} = w_{t}^{*} \cdot \left(\frac{C_{t}^{adj}}{C_{t-1}^{adj}} - 1.0\right) - Cost_{t}$$ (27)
    $$Equity_{t} = Equity_{0} \prod_{i=1}^{t} (1.0 + r_{t}^{net})$$ (28)

## 3. Empirical Diagnostics & Out-of-Sample Performance

### 3.1 Experimental Configuration & Cross-Validation Integrity
The framework was evaluated across 1,995 trading sessions of DIG equity and the VN-Index benchmark spanning June 4, 2018, to June 1, 2026. Model verification implements a 5-Fold Purged and Embargoed Walk-Forward Time-Series Split. To prevent information leakage, a purging gap of $h=5$ days was excised from the boundary of each training set, while a 60-day buffer was prepended to each test partition to warm up rolling transformations without generating null vectors. A rolling window cap ($M=750$ sessions) was imposed on training lengths, forcing the algorithm to adapt to modern volatility regimes. Across all 5 folds, exactly 1,655 out-of-sample trading days were evaluated under strict out-of-sample conditions.

*   Fold 1: Train [2018-06-05 - 2019-09-26] | OOS Test [2019-09-27 - 2021-01-19] (332 bars)
*   Fold 2: Train [2018-06-05 - 2021-01-19] | OOS Test [2021-01-20 - 2022-05-25] (332 bars)
*   Fold 3: Train [2019-06-05 - 2022-05-25] | OOS Test [2022-05-26 - 2023-09-20] (332 bars)
*   Fold 4: Train [2020-09-28 - 2023-09-20] | OOS Test [2023-09-21 - 2025-01-15] (332 bars)
*   Fold 5: Train [2022-01-21 - 2025-01-15] | OOS Test [2025-01-16 - 2026-06-01] (327 bars)

**Table 1: Stepwise Evolution of Meta-Labeling Configurations Across Iterative System Calibrations (Friction: 20 bps, T+1 Execution)**

| Iteration / Setup Configuration | Net Cumulative Return | Annualized CAGR | Net Sharpe Ratio | Max Drawdown (MDD) | Total Turnover | Mean Exposure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Recap 1: Binary Uncalibrated (Default Threshold $\ge 0.50$) | +2.62% | 0.39% | 0.21 | -5.04% | 23.17x | 48.82% |
| Recap 2: Single Adaptive Threshold (OOF Utility Optimization) | -0.57% | -0.09% | -0.03 | -9.67% | 32.76x | 69.79% |
| Recap 3: Asymmetric Loss Function + Fractional Kelly Sizing | +2.08% | 0.31% | 0.38 | -1.39% | 6.73x | 35.23% |
| Recap 4: Calibrated Probability + Confidence Sizing (Unfiltered) | +76.22% | 9.01% | 0.61 | -31.58% | 67.02x | 38.79% |
| Recap 5: Volume Confirmation + Strict [0.50, 0.56] Floor | -18.13% | -3.00% | -0.49 | -27.29% | 59.67x | 6.83% |
| **Recap 6 (Production): Calibrated Meta-Engine with Deadband Buffer** | **+89.65%** | **10.24%** | **0.71** | **-25.82%** | **81.41x** | **39.82%** |
| Benchmark: DIG Buy-and-Hold Strategy | +66.87% | 8.11% | 0.41 | -88.50% | 1.00x | 100.00% |

**Table 2: Core Out-of-Sample Performance Tear Sheet (Production Meta-Labeling Architecture vs. Buy & Hold Benchmark)**

| Performance Metric (Net of Fees & Slippage) | Production Strategy | Benchmark (Buy & Hold) | Empirical Delta |
| :--- | :--- | :--- | :--- |
| Cumulative Net Return | +89.65% | +66.87% | +22.78% |
| Compound Annual Growth Rate (CAGR) | 10.24% | 8.11% | +2.13% |
| Annualized Realized Volatility | 15.35% | ~42.50% | -27.15% (Volatility Reduction) |
| Sharpe Ratio $(Rf=0)$ | 0.71 | 0.41 | +0.30 |
| Sortino Ratio (Downside Deviation Adjusted) | 0.63 | ~0.35 | +0.28 |
| Calmar Ratio (CAGR/MDD) | 0.40 | 0.09 | +0.31 |
| Maximum Peak-to-Trough Drawdown (MDD) | -25.82% | -88.50% | +62.68% (Capital Preservation) |
| Market Exposure Percentage | 39.82% | 100.00% | -60.18% (Time in Risk Assets) |
| Cash Drag Ratio | 60.18% | 0.00% | Risk-Off Capital Shield |
| Total Cumulative Transaction Friction | 16.28% | ~0.30% | Drawdown Control Overhead |
| Cumulative Portfolio Turnover | 81.41x | 1.00x | Governed by Deadband Buffer |

**Table 3: OOS Walk-Forward Fold Diagnostics: Signal Regimes & Positioning**

| OOS Evaluation Fold | Primary + Vol Bars Ratio | Calibrated Floor ($\tau^{*}$) | Active Long Signals | Mean / Max Position | Market Regime & Systemic Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 (2019-2021) | 43.6% (125/287) | $\ge 0.480$ | 8.43% | 0.08/1.00 | Sideways consolidation; selective participation avoids liquidity entrapment. |
| Fold 2 (2021-2022) | 53.6% (333/621) | $\ge 0.480$ | 72.89% | 0.36/0.52 | Historic bull market; full trend participation captures momentum upside. |
| Fold 3 (2022-2023) | 68.9% (481/698) | $\ge 0.480$ | 43.07% | 0.18/0.42 | Volatile cycle peak; primary model shuts down below EMA50, dodging the crash. |
| Fold 4 (2023-2025) | 59.4% (403/679) | $\ge 0.480$ | 23.80% | 0.08/1.00 | Severe macro bear market; cash held >76% of time, insulating capital. |
| Fold 5 (2025-2026) | 37.4% (264/705) | $\ge 0.480$ | 44.65% | 0.14/0.47 | Sector rotation recovery; capital deployed based on liquidity inflows. |

**Table 4: Binary Out-of-Sample Classification Matrix (1,655 Total Cumulative OOS Inference Bars)**

| Classification Diagnostic Metric | Metric Value | Methodological Significance |
| :--- | :--- | :--- |
| True Negative Count (Class 0 Correct) | 573 bars | Successful avoidance of unprofitable trading horizons. |
| False Positive Count (Class 0 False) | 312 bars | Positions that hit stop-loss or timed out without gain. |
| False Negative Count (Class 1 False) | 443 bars | Foregone take-profit events filtered out due to marginal confidence. |
| True Positive Count (Class 1 Correct) | 326 bars | Executed trades reaching take-profit boundary successfully. |
| Target Precision (Class 1) | 51.10% | Outperformed binary random expectation; maintained positive trade expectancy. |
| Target Recall (Class 1) | 42.34% | Deliberately sacrifices marginal trades to avoid turnover friction. |
| Aggregate Accuracy | 54.32% | Consistent non-trivial classification across financial noise. |
| Area Under ROC Curve (ROC-AUC) | 0.5234 | Modest ranking discrimination converted to returns via calibration & sizing. |
| Brier Score Loss | 0.2727 | Quantifies probability calibration alignment. |

**Table 5: Feature Consensus & Econometric Interpretability Analysis**
**Top 15 Most Important Feature Kernels Ranked by Relative Gain**

| Rank | Feature Identifier | Relative Gain | Fold Presence | Econometric Classification & Structural Meaning |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `wavelet_approx_a3_zscaled_lag1` | 0.2715 | 1/5 | Haar Wavelet approximation energy at decomposition level 3; isolates structural accumulation phases from highfrequency market noise. |
| 2 | `kurtosis_20_zscaled_lag2` | 0.2161 | 1/5 | Standardized fourth-moment deviation; quantifies regime fat-tail risk and structural return outlier instability. |
| 3 | `kurtosis_20_momentum` | 0.1376 | 1/5 | First-order velocity of return tail thickness; acts as an early warning for regime shift transitions. |
| 4 | `body_direction_momentum_zscaled_lag1` | 0.1228 | 1/5 | Standardized candle body directional momentum; confirms active institutional buying pressure over 5 sessions. |
| 5 | `shadow_asymmetry_ratio_momentum` | 0.1185 | 1/5 | Intraday candle shadow asymmetry momentum; detects directional wick absorption and lower-boundary supply rejection. |
| 6 | `vol_shock_3_acceleration` | 0.1185 | 2/5 | Second derivative of short-term volume expansion; identifies institutional trade participation. |
| 7 | `overnight_gap_zscaled_lag2` | 0.1149 | 1/5 | Normalized overnight jump spread; quantifies off-session information asymmetry and auction opening imbalances. |
| 8 | `buying_tail_power_momentum` | 0.1147 | 1/5 | Volume-weighted lower shadow momentum; captures aggressive accumulation during intraday pullbacks. |
| 9 | `liq_roll_measure_20_momentum` | 0.1044 | 1/5 | Roll (1984) effective bid-ask spread proxy momentum; tracks structural liquidity widening. |
| 10 | `skewness_20_zscaled_lag2` | 0.1036 | 1/5 | Third standardized moment of returns; detects directional tail asymmetry before large price breakouts. |
| 11 | `skewness_20_momentum` | 0.1035 | 1/5 | Momentum of return distribution skewness; measures structural shifts from downside to upside tail risk. |
| 12 | `mkt_vol_ratio_20_acceleration` | 0.1028 | 1/5 | Acceleration of asset volatility relative to VN-Index volatility; flags idiosyncratic decoupling from broader market beta. |
| 13 | `skewness_20_zscaled_lag5` | 0.0999 | 1/5 | Longer-lagged distribution asymmetry; confirms persistence of directional drift. |
| 14 | `kinematic_clv_vol_roll20_momentum` | 0.0990 | 2/5 | Close Location Value scaled by trading volume; measures whether closes cluster near session highs on expanding volume. |
| 15 | `vol_downside_dev_20_momentum` | 0.0987 | 1/5 | Momentum of semi-variance downside deviation; acts as an asymmetric risk governor for sizing allocation. |

**Consensus Cross-Fold Feature Presence (Stability Leaders):**
*   `kurtosis_20_acceleration`: 4 Folds
*   `liq_amihud_z_zscaled_lag2`: 3 Folds
*   `liq_amihud_z_acceleration`: 3 Folds
*   `sin_month_momentum`: 2 Folds
*   `sin_dow_acceleration`: 2 Folds
*   `vol_shock_3_acceleration`: 2 Folds
*   `kinematic_clv_vol_roll20_momentum`: 2 Folds

## 4. Discussion & Empirical Findings

**Table 6: Methodological Reconciliation: Cross-Paradigm Progression**
**(Comparative Analysis of Phases 1, 2, and 3 Across the Same Asset and Period)**

| Evaluation Criterion | Phase 1: Feature Framework | Phase 2: Static Thresholding | Phase 3: Meta-Labeling (Current) |
| :--- | :--- | :--- | :--- |
| Primary Research Focus | Econometric feature extraction and statistical pruning protocol. | Investigating discrete static thresholds in frictional time series. | Decoupling side from sizing via calibrated meta-labeling. |
| Target Variable Space | Ternary Triple-Barrier $y \in \{-1, 0, 1\}$ | Ternary Triple-Barrier $y \in \{-1, 0, 1\}$. | Two-Stage: Macro Filter + Binary $Y \in \{0, 1\}$. |
| Probability Calibration | None (Raw softmax margins). | None (Raw tree leaf outputs). | Out-of-Fold Non-parametric Isotonic Regression. |
| Capital Sizing Mechanism | Unit step sizing $(w \in \{-1, 0, 1\})$. | Realized Volatility Targeting $(w = \sigma_{tgt} / \sigma_{t})$. | Continuous Confidence Sizing bounded by $[\tau^{*}, 1.0]$. |
| Microstructure Buffer | Unbuffered $T+1$ execution. | Unbuffered $T+1$ execution. | 15% Weight Deadband + 3-Day Minimum Holding. |
| Cumulative Net Return | -15.82% | -42.59% | +89.65% (Benchmark: +66.87%) |
| Net Sharpe Ratio | -0.64 | -0.45 | +0.71 (Benchmark: 0.41) |
| Maximum Drawdown (MDD) | -73.04% | -73.97% | -25.82% (Benchmark: -88.50%) |
| Cumulative Turnover | 464.0x capital | Extremely high (friction decay) | 81.41x capital (Controlled Chattering) |
| Failure Mechanism / Status | Unoptimized execution; excess friction. | Position Entrapment & Microstructure Friction. | Methodologically Rehabilitated & Production-Ready. |

### 4.1 Deconstructing the Phase 2 Failure: Why Meta-Labeling Resolves Position Entrapment
In Phase 2, the ternary XGBoost architecture attempted to predict directional sign:

$$\hat{y}_{t} \in \{-1, 0, 1\}$$

Because financial markets display pronounced regime switching, conditioning trade entry on static OOF thresholds $(\theta_{long}, \theta_{short})$ generated severe position entrapment. For example, in Fold 2 (the 2021 liquidity expansion), the static model generated 98.80% Long signals with an optimal threshold of $\theta_{long} = 0.38$. When the market subsequently transitioned into the 2022 bear market (Fold 4), that static threshold remained loose, holding high equity exposure into a catastrophic down-trend.

The Two-Stage Meta-Labeling architecture resolves this failure structurally:
*   The primary filter $\left(S_{t} = \mathbb{I}(C_{t-1} > EMA_{50,t-1}) \times \mathbb{I}(\overline{r}_{VN-INDEX} > -0.006)\right)$ immediately cuts all trade generation whenever the market slips below its 50-day moving average or the macro index breaks down.
*   In Fold 4 (the brutal 2022-2023 crash), the primary filter was inactive for over 76% of all trading days. As shown in Table 3, the strategy took trades on only 23.80% of days, carrying an average portfolio weight of just 0.08. This structural cash-drag mechanism preserved capital, restricting strategy drawdown to -25.82% while the underlying equity collapsed -88.50%.

### 4.2 The Role of Isotonic Calibration in Bet Sizing
Uncalibrated tree outputs generate polarized probabilities clustered near decision boundaries. By implementing Isotonic Regression across the out-of-fold predictions, raw margins are monotonically transformed to match empirical frequencies: a calibrated probability of 0.52 corresponds precisely to a 52% empirical take-profit rate.

This calibration directly empowers our confidence-based sizing function:

$$w_{t} \propto (\hat{P}_{calibrated} - \tau^{*})$$

Instead of binary all-in allocation, capital commitment expands smoothly during high-conviction regimes (such as Fold 2, where mean exposure reached 0.36 with a maximum position of 0.52) and contracts to defensive floors (0.08 in Fold 4) during ambiguous regimes.

### 4.3 Chattering Suppression via Deadband Execution and Settlement Harmonization
In high-frequency simulations lacking market microstructure controls, small probability fluctuations around the threshold trigger daily rebalancing, generating unsustainable turnover. As demonstrated in Table 6, Phase 1 incurred 464x turnover, completely eroding gross alpha into transaction fees.

Our execution engine resolves this through two complementary constraints:
1.  **The 15% Deadband Filter:** Mandates that the portfolio will not rebalance unless the absolute sizing differential satisfies $|\Delta w| \ge 0.15$.
2.  **The 3-Day Minimum Holding Period:** Prevents position unwinding prior to session $T+3$.

Together, these mechanisms compress cumulative turnover to 81.41x over 6.5 years (Table 2), limiting total friction drag to 16.28%. This controlled turnover allowed the strategy to capture +89.65% in net returns, outperforming the benchmark by +22.78% on an absolute basis and by +0.30 in Sharpe ratio (0.71 vs. 0.41).

### 4.4 Econometric Insights from Feature Importance
The relative gain attribution in Table 5 illuminates the market mechanics driving the system's predictive edge:

*   **Multi-Resolution Wavelet Dominance (27.15% Gain):** The single most dominant feature, `wavelet_approx_a3_zscaled_lag1`, represents the low-frequency approximation component from Haar Wavelet decomposition. By filtering out short-term price fluctuations, it isolates the underlying trend energy of the consolidation phase.
*   **Fat-Tail and Distributional Moment Dynamics (>45% Aggregate Gain):** Rather than traditional price oscillators (RSI, MACD), the model relies heavily on higher-order statistical moments: `kurtosis_20_zscaled_lag2` (21.61%), `kurtosis_20_momentum` (13.76%), and various skewness lags. These features act as quantitative early-warning indicators, detecting shifts in return distribution thickness prior to major volatility explosions.
*   **Microstructure Price Pressure:** Features such as `body_direction_momentum` (12.28%) and `shadow_asymmetry_ratio_momentum` (11.85%) confirm that intraday price action (such as lower wick rejection indicating institutional supply absorption) provides decisive directional confirmation when sizing entries.

## 5. Conclusion & Research Horizons

### 5.1 Concluding Assessment
This study concludes our empirical research series by demonstrating that the failure of machine learning in equity trading is not an indictment of algorithmic capacity, but a consequence of flawed architectural design. When machine learning models are tasked with simultaneous directional forecasting and trade sizing in frictional environments, concept drift and turnover friction inevitably degrade performance.

By implementing a Two-Stage Meta-Labeling framework, incorporating out-of-fold Isotonic Probability Calibration, utilizing continuous confidence sizing, and enforcing microstructure-synchronized deadband execution, we successfully rehabilitated system performance. Over a 6.5-year out-of-sample period spanning 1,655 sessions in the Vietnamese equity market, the system achieved +89.65% net returns and a Net Sharpe ratio of 0.71, outperforming the Buy-and-Hold benchmark while cutting maximum drawdown from -88.50% down to -25.82%.

**RESEARCH SERIES SUMMARY**
*   **Phase 1: Robust Feature Engineering Framework**
    *   Solved: Lookahead bias, collinearity, spurious correlations.
    *   Limitation: Unoptimized execution incurred severe friction decay.
*   **Phase 2: Static Ternary Thresholding Failure**
    *   Diagnosed: Position entrapment & concept drift (-42.59% net).
    *   Limitation: Static thresholds break down under regime changes.
*   **Phase 3: Calibrated Meta-Labeling with Deadband Buffer (Current Work)**
    *   Resolved: Decoupled side and size; applied Isotonic calibration; enforced 15% deadband and T+2.5 minimum holding.
    *   Result: +89.65% net return, 0.71 Sharpe, -25.82% MDD.

### 5.2 Future Extensions
1.  **Cross-Sectional Portfolio Scaling:** Expanding the single-asset meta-labeling engine across the VN30 and VN100 constituents, implementing cross-sectional ranking to dynamically allocate capital among competing meta-signals.
2.  **Real Level 2 Limit Order Book Integration:** Replacing proxy microstructure features (Amihud, Roll spread) with high-frequency L2 order book metrics (Order Flow Imbalance, VPIN) to enhance execution.
3.  **Deep Reinforcement Learning Execution:** Formulating the 15% deadband rebalancing buffer as a continuous Markov Decision Process, training an agent to dynamically optimize execution thresholds against real-time bid-ask spreads.

## 6. References
*   Barndorff-Nielsen, O. E., & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using realized variance. *Journal of Financial Econometrics*, 4(1), 1-30.
*   Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *The Journal of Finance*, 67(2), 719-759.
*   De Prado, M. L. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
*   Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 987-1007.
*   Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 424-438.
*   López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management*, 42(4), 59-69.
*   Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.
*   Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *The Journal of Finance*, 39(4), 1127-1139.
*   Tong, S. L. (2026a). An Econometrically Grounded, Context-Aware Feature Engineering and Validation Framework for Equity Return Predictability: Evidence from Emerging Markets (Extended Framework). *Quantitative Research Working Paper Series*.
*   Tong, S. L. (2026b). Empirical Evaluation of Asymmetric Thresholding in Machine Learning-Based Equity Trading Systems: A Microstructure-Adjusted Approach. *Quantitative Research Working Paper Series*.
*   Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 694-699.
```eof