import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from .calibration import (
    StandardPurgedCV,
    compute_meta_confidence_sizing,
    extract_primary_signals,
    optimize_meta_threshold,
)
from .feature_selection import EconometricsFeaturePipeline


class ProductionMetaTrainer:
    """
    Walk-forward Purged CV Meta-Labeling orchestrator with Optuna and Isotonic calibration.
    """

    def __init__(
        self,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        target_horizon: int = 5,
        buffer_days: int = 60,
        max_train_size: int = 750,
    ):
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.target_horizon = target_horizon
        self.buffer_days = buffer_days
        self.max_train_size = max_train_size

    def run_training_pipeline(self, df_final: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
        df = df_final.copy()
        df["primary_signal"] = extract_primary_signals(df)

        outer_cv = StandardPurgedCV(
            n_splits=self.n_outer_splits,
            purge_gap=self.target_horizon,
            max_train_size=self.max_train_size,
        )

        oos_predictions = []
        trained_models = []
        selected_features_per_fold = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df)):
            print(f"\n>>> [FOLD {fold + 1}/{self.n_outer_splits}]")
            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # Zero-variance check
            numeric_cols = df_train_raw.select_dtypes(include=[np.number]).columns
            zero_cols = [c for c in numeric_cols if df_train_raw[c].std(ddof=0) < 1e-8]
            if zero_cols:
                df_train_raw.drop(columns=zero_cols, inplace=True)
                df_test_raw.drop(columns=zero_cols, inplace=True, errors="ignore")

            pipeline = EconometricsFeaturePipeline(target_horizon=self.target_horizon)
            pipeline.fit(df_train_raw)

            current_features = list(pipeline.final_features_list_)
            selected_features_per_fold.append(set(current_features))

            X_train_transformed = pipeline.transform(df_train_raw)
            _, y_train_series = pipeline._prepare_target(df_train_raw)

            common_train_idx = X_train_transformed.index.intersection(y_train_series.index)
            X_tr_all = X_train_transformed.loc[common_train_idx, current_features].dropna()
            y_tr_all = y_train_series.loc[X_tr_all.index]

            primary_train_mask = df_train_raw.loc[X_tr_all.index, "primary_signal"] == 1
            X_tr = X_tr_all.loc[primary_train_mask]
            y_tr_mapped = y_tr_all.loc[primary_train_mask].astype(int).values

            if len(np.unique(y_tr_mapped)) < 2:
                X_tr = X_tr_all
                y_tr_mapped = y_tr_all.astype(int).values

            n_zeros = np.sum(y_tr_mapped == 0)
            n_ones = np.sum(y_tr_mapped == 1)
            scale_pos_weight_val = float(n_zeros) / (n_ones + 1e-8)
            fold_sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr_mapped)

            inner_cv = StandardPurgedCV(n_splits=self.n_inner_splits, purge_gap=self.target_horizon)

            def objective(trial):
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "random_state": 42,
                    "n_jobs": 2,
                    "scale_pos_weight": trial.suggest_float(
                        "scale_pos_weight", 0.8 * scale_pos_weight_val, 1.2 * scale_pos_weight_val
                    ),
                    "n_estimators": trial.suggest_int("n_estimators", 40, 100, step=20),
                    "max_depth": trial.suggest_int("max_depth", 2, 4),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
                    "subsample": trial.suggest_float("subsample", 0.65, 0.85),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.85),
                    "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 25.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 4, 12),
                }

                val_losses = []
                for in_tr_idx, in_val_idx in inner_cv.split(X_tr):
                    if len(np.unique(y_tr_mapped[in_tr_idx])) < 2:
                        continue
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_tr.iloc[in_tr_idx],
                        y_tr_mapped[in_tr_idx],
                        sample_weight=fold_sample_weights[in_tr_idx],
                        verbose=False,
                    )
                    probs = model.predict_proba(X_tr.iloc[in_val_idx])[:, 1]
                    val_losses.append(
                        log_loss(
                            y_tr_mapped[in_val_idx],
                            probs,
                            sample_weight=fold_sample_weights[in_val_idx],
                            labels=[0, 1],
                        )
                    )
                return np.mean(val_losses) if val_losses else 1.0

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10, timeout=90)

            best_params = study.best_params.copy()
            best_params.update({
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            })

            # OOF estimation for Isotonic Calibration
            oof_raw_probs = np.zeros(len(X_tr))
            oof_counts = np.zeros(len(X_tr))

            for in_tr_idx, in_val_idx in inner_cv.split(X_tr):
                fold_model = xgb.XGBClassifier(**best_params)
                fold_model.fit(
                    X_tr.iloc[in_tr_idx],
                    y_tr_mapped[in_tr_idx],
                    sample_weight=fold_sample_weights[in_tr_idx],
                    verbose=False,
                )
                oof_raw_probs[in_val_idx] += fold_model.predict_proba(X_tr.iloc[in_val_idx])[:, 1]
                oof_counts[in_val_idx] += 1

            valid_mask = oof_counts > 0
            oof_raw_probs = oof_raw_probs[valid_mask] / oof_counts[valid_mask]
            oof_true = y_tr_mapped[valid_mask]

            iso_calibrator = IsotonicRegression(out_of_bounds="clip")
            iso_calibrator.fit(oof_raw_probs, oof_true)
            oof_calibrated_probs = iso_calibrator.predict(oof_raw_probs)

            min_req_trades = max(10, int(len(oof_true) * 0.08))
            meta_threshold, oof_utility = optimize_meta_threshold(
                oof_probs=oof_calibrated_probs,
                oof_true=oof_true,
                min_trades=min_req_trades,
                min_coverage=0.08,
                beta=0.5,
            )

            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(X_tr, y_tr_mapped, sample_weight=fold_sample_weights, verbose=False)
            trained_models.append(final_model)

            # Inference on pure OOS partition
            df_test_buffer = pd.concat([df_train_raw.iloc[-self.buffer_days :], df_test_raw])
            X_test_transformed = pipeline.transform(df_test_buffer)
            X_test_pure = X_test_transformed.loc[X_test_transformed.index >= df_test_raw.index.min()]
            _, y_test_series = pipeline._prepare_target(df_test_raw)

            common_test_idx = X_test_pure.index.intersection(y_test_series.index)
            X_te = X_test_pure.loc[common_test_idx].reindex(columns=current_features, fill_value=0.0)
            y_te = y_test_series.loc[common_test_idx]
            test_primary_signals = df_test_raw.loc[common_test_idx, "primary_signal"].values

            raw_test_probs = final_model.predict_proba(X_te)[:, 1]
            test_calibrated_probs = iso_calibrator.predict(raw_test_probs)

            active_sizing = compute_meta_confidence_sizing(
                calibrated_probs=test_calibrated_probs,
                meta_thresh=meta_threshold,
                base_min_weight=0.25,
            )
            final_bet_sizes = np.where(test_primary_signals == 1, active_sizing, 0.0)
            y_pred_binary = (final_bet_sizes > 0.0).astype(float)

            res_df = pd.DataFrame(
                {
                    "fold": fold + 1,
                    "primary_signal": test_primary_signals,
                    "prob_raw": raw_test_probs,
                    "prob_success": test_calibrated_probs,
                    "meta_threshold": meta_threshold,
                    "model_raw_target": y_pred_binary,
                    "bet_size": final_bet_sizes,
                    "true_target": y_te.values,
                },
                index=common_test_idx,
            )
            oos_predictions.append(res_df)

        df_oos_calibrated = pd.concat(oos_predictions).sort_index()
        return df_oos_calibrated, trained_models, selected_features_per_fold