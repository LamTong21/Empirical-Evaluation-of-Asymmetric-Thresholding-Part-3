import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import grangercausalitytests

from .econometric_engine import (
    Layer10FeatureRouter,
    Stage2DiagnosticEngine,
    Stage3FeatureEngine,
)


class Layer12RedundancyControl:
    @staticmethod
    def linear_vif_prune(X: pd.DataFrame, threshold: float = 0.85, max_vif: float = 5.0) -> pd.DataFrame:
        X_curr = X.copy()
        corr_matrix = X_curr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(upper[col] > threshold)]
        X_curr = X_curr.drop(columns=to_drop_corr)

        while True:
            if X_curr.shape[1] <= 1:
                break
            vifs = []
            cols = X_curr.columns
            X_vals = X_curr.values
            for j in range(len(cols)):
                y_target = X_vals[:, j]
                X_other = np.delete(X_vals, j, axis=1)
                X_mat = np.column_stack([np.ones(len(X_other)), X_other])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_mat, y_target, rcond=None)
                    preds = X_mat @ beta
                    ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
                    ss_res = np.sum((y_target - preds) ** 2)
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
                    vif = 1.0 / (1.0 - np.clip(r2, 0, 0.9999))
                except np.linalg.LinAlgError:
                    vif = max_vif + 1.0
                vifs.append(vif)

            max_vif_val = max(vifs)
            if max_vif_val > max_vif:
                drop_idx = np.argmax(vifs)
                X_curr = X_curr.drop(columns=[cols[drop_idx]])
            else:
                break
        return X_curr

    @staticmethod
    def nonlinear_hrp_prune(X: pd.DataFrame, max_clusters: int = 15) -> pd.DataFrame:
        if X.shape[1] <= max_clusters:
            return X
        corr = X.corr(method="spearman").clip(-1.0, 1.0).fillna(0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))
        dist_sym = (dist + dist.T) / 2.0
        np.fill_diagonal(dist_sym.values, 0)

        condensed_dist = squareform(dist_sym, checks=False)
        link = linkage(condensed_dist, method="ward")
        clusters = fcluster(link, t=max_clusters, criterion="maxclust")

        selected_medoids = []
        for c_id in np.unique(clusters):
            cluster_cols = X.columns[clusters == c_id]
            if len(cluster_cols) == 1:
                selected_medoids.append(cluster_cols[0])
            else:
                sub_dist = dist_sym.loc[cluster_cols, cluster_cols]
                medoid = sub_dist.sum(axis=1).idxmin()
                selected_medoids.append(medoid)
        return X[selected_medoids]


class Layer13PredictiveDiagnostics:
    @staticmethod
    def causality_screen(X: pd.DataFrame, y: pd.Series, max_lag: int = 5, p_threshold: float = 0.05) -> tuple:
        valid_features = []
        optimal_lags = {}
        df_test = pd.concat([y, X], axis=1).dropna()
        if df_test.empty:
            return valid_features, optimal_lags

        y_col = df_test.columns[0]
        y_vals = df_test[y_col].values

        for col in X.columns:
            test_data = df_test[[y_col, col]]
            best_lag = 1
            is_causal = False
            best_p = 1.0

            try:
                gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_val = gc_res[lag][0]["ssr_ftest"][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag
                if best_p < p_threshold:
                    is_causal = True
            except Exception:
                pass

            if not is_causal:
                best_mi = 0.0
                for lag in range(1, max_lag + 1):
                    x_lagged = df_test[col].shift(lag).fillna(0).values
                    mi_score = mutual_info_classif(x_lagged.reshape(-1, 1), y_vals, random_state=42)[0]
                    if mi_score > best_mi:
                        best_mi = mi_score
                        best_lag = lag
                if best_mi > 0.01:
                    is_causal = True

            if is_causal:
                valid_features.append(col)
                optimal_lags[col] = best_lag

        if len(valid_features) < 3:
            corr_with_y = df_test.drop(columns=[y_col]).corrwith(df_test[y_col], method="spearman").abs()
            top_fallback = corr_with_y.sort_values(ascending=False).head(5).index.tolist()
            for f in top_fallback:
                if f not in valid_features:
                    valid_features.append(f)
                    optimal_lags[f] = 1

        return valid_features, optimal_lags


class Layer14LagTransformEngine:
    @staticmethod
    def apply_volatility_scaled_lags(X: pd.DataFrame, optimal_lags: dict, eps: float = 1e-8) -> pd.DataFrame:
        X_transformed = pd.DataFrame(index=X.index)
        for col, lag in optimal_lags.items():
            if col not in X.columns:
                continue
            feature_series = X[col]
            feature_lagged = feature_series.shift(lag)
            roll_mean = feature_lagged.rolling(20).mean()
            roll_std = feature_lagged.rolling(20).std(ddof=1)

            z_scaled = (feature_lagged - roll_mean) / (roll_std + eps)
            X_transformed[f"{col}_zscaled_lag{lag}"] = z_scaled
            momentum = (feature_series - feature_lagged) / (feature_lagged.abs() + eps)
            acceleration = momentum - momentum.shift(1)
            X_transformed[f"{col}_momentum"] = momentum
            X_transformed[f"{col}_acceleration"] = acceleration

        return X_transformed


class Stage4SelectionRouter:
    def __init__(self, payload: dict):
        self.payload = payload
        self.max_clusters = payload.get("max_clusters", 14)
        self.mi_threshold = payload.get("mi_threshold", 0.01)
        self.min_features = 6
        self.max_features = 16
        self.directional_priority_keywords = [
            "mkt_relative", "mkt_rs", "mkt_rolling_beta", "mkt_divergence",
            "cmf", "obv", "vpt", "net_pressure", "vol_directional",
            "signed_expansion", "liq_amihud", "kurtosis_20"
        ]

    def execute(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        aligned = pd.concat([X_train, y_train], axis=1).dropna()
        if aligned.empty:
            return X_train, {}, list(X_train.columns)

        X_mat = aligned.iloc[:, :-1]
        y_mat = aligned.iloc[:, -1]

        zero_cols = [c for c in X_mat.columns if X_mat[c].std() < 1e-8]
        if zero_cols:
            X_mat = X_mat.drop(columns=zero_cols)

        X_pruned = Layer12RedundancyControl.linear_vif_prune(X_mat, threshold=0.85, max_vif=6.0)
        X_hrp = Layer12RedundancyControl.nonlinear_hrp_prune(X_pruned, max_clusters=self.max_clusters)

        valid_causal_feats, optimal_lags = Layer13PredictiveDiagnostics.causality_screen(X_hrp, y_mat)
        X_causal_subset = X_hrp[[col for col in valid_causal_feats if col in X_hrp.columns]]
        X_transformed = Layer14LagTransformEngine.apply_volatility_scaled_lags(X_causal_subset, optimal_lags)

        X_transformed_clean = X_transformed.dropna()
        y_mat_clean = y_mat.loc[X_transformed_clean.index]

        valid_trans_cols = [c for c in X_transformed_clean.columns if X_transformed_clean[c].std() > 1e-8]
        X_transformed_clean = X_transformed_clean[valid_trans_cols]

        mi_scores = mutual_info_classif(X_transformed_clean, y_mat_clean, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_transformed_clean.columns).sort_values(ascending=False)

        is_priority = mi_series.index.map(lambda name: any(k in name for k in self.directional_priority_keywords))
        priority_features = mi_series[is_priority & (mi_series > (self.mi_threshold * 0.5))].index.tolist()
        general_candidates = mi_series[mi_series > self.mi_threshold].index.tolist()

        combined_selected = []
        for feat in priority_features + general_candidates:
            if feat not in combined_selected:
                combined_selected.append(feat)

        if len(combined_selected) < self.min_features:
            valid_features = mi_series.head(self.min_features).index.tolist()
        elif len(combined_selected) > self.max_features:
            valid_features = combined_selected[: self.max_features]
        else:
            valid_features = combined_selected

        return X_transformed[valid_features], optimal_lags, valid_features


class Stage4SelectionEngine:
    def __init__(self, payload: dict):
        self.payload = payload
        self.router = Stage4SelectionRouter(payload)
        self.optimal_lags_ = None
        self.selected_features_ = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        _, optimal_lags, final_features = self.router.execute(X_train, y_train)
        self.optimal_lags_ = optimal_lags
        self.selected_features_ = final_features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.optimal_lags_ is None or self.selected_features_ is None:
            raise ValueError("Mô hình chưa được fit(). Vui lòng gọi fit() trên tập train trước.")
        X_transformed = Layer14LagTransformEngine.apply_volatility_scaled_lags(X, self.optimal_lags_)
        available_cols = [c for c in self.selected_features_ if c in X_transformed.columns]
        return X_transformed[available_cols]


class EconometricsFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end econometric feature engine preserving time barriers between folds.
    """

    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon
        self.is_fitted = False
        self.s2_engine = None
        self.s3_engine = None
        self.s4_engine = None
        self.routing_payload_ = None
        self.final_features_list_ = None

    def _prepare_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = df["target_label"].copy().rename("target")
        valid_idx = y.dropna().index
        return df.loc[valid_idx], y.loc[valid_idx]

    def fit(self, X: pd.DataFrame, y=None):
        X_data, y_data = self._prepare_target(X)
        aligned = pd.concat([X_data, y_data], axis=1).dropna()
        X_train_aligned = aligned.drop(columns=["target"])
        y_train_aligned = aligned["target"]

        self.s2_engine = Stage2DiagnosticEngine()
        self.routing_payload_ = self.s2_engine.fit(X_train_aligned)

        self.s3_engine = Stage3FeatureEngine(self.routing_payload_)
        X_train_candidates = self.s3_engine.transform(X_train_aligned)

        aligned_s3 = pd.concat([X_train_candidates, y_train_aligned], axis=1).dropna()
        X_train_candidates = aligned_s3.drop(columns=["target"])
        y_train_candidates = aligned_s3["target"]

        self.s4_engine = Stage4SelectionEngine(self.routing_payload_)
        self.s4_engine.fit(X_train_candidates, y_train_candidates)

        self.final_features_list_ = self.s4_engine.selected_features_
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit. Hãy gọi fit(df_train) trước.")
        X_test_candidates = self.s3_engine.transform(X)
        return self.s4_engine.transform(X_test_candidates)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)