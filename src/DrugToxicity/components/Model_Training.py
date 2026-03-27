import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from src.DrugToxicity.utils.common import logger
from src.DrugToxicity.entity.config_entity import ModelTrainerConfig


TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

## Model factory functions
def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42
    )

def make_xgb(spw: int):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0
    )

def make_lgbm(spw: int):
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        num_leaves=63, min_child_samples=10,
        n_jobs=-1, random_state=42, verbose=-1
    )

def make_meta():
    return LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000
    )


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        Skips if model_bundle_path already exists.
        Otherwise trains XGBoost + RandomForest + LightGBM stacking ensemble:
          - Per-target scale_pos_weight (neg // pos) handles class imbalance
          - 5-fold OOF predictions train the LogisticRegression meta-learner
          - Base models retrained on full training split before saving
          - X_te.npy + y_te.npy saved for model evaluation stage
        Returns True if trained, False if skipped.
        """
        if self.config.model_bundle_path.exists():
            logger.info(
                f"Model bundle already exists at {self.config.model_bundle_path} "
                "— skipping training."
            )
            return False

        os.makedirs(self.config.root_dir, exist_ok=True)

        ## Load features and targets
        X = np.load(self.config.transformed_data_dir / "X.npy")
        y = np.load(self.config.transformed_data_dir / "y.npy")
        logger.info(f"Loaded X{X.shape} y{y.shape}")

        ## Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Train: {X_tr.shape} | Test: {X_te.shape}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_bundle = {}

        for i, target in enumerate(TARGET_COLS):
            mask_tr = y_tr[:, i] != -1
            X_t = X_tr[mask_tr]
            y_t = y_tr[mask_tr, i]

            if len(np.unique(y_t)) < 2 or len(y_t) < 50:
                logger.warning(f"  {target}: skipped (insufficient labelled data).")
                continue

            pos = int(y_t.sum())
            neg = int((y_t == 0).sum())
            spw = max(1, neg // max(1, pos))   ## per-target imbalance ratio

            logger.info(f"  {target}: {len(y_t)} samples | pos={pos} neg={neg} spw={spw}")

            oof_rf   = np.zeros(len(y_t))
            oof_xgb  = np.zeros(len(y_t))
            oof_lgbm = np.zeros(len(y_t))

            ## 5-fold OOF for meta-learner (no leakage)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_t, y_t)):
                Xf_tr, Xf_val = X_t[tr_idx], X_t[val_idx]
                yf_tr         = y_t[tr_idx]

                _rf = make_rf();       _rf.fit(Xf_tr, yf_tr)
                _xg = make_xgb(spw);  _xg.fit(Xf_tr, yf_tr)
                _lg = make_lgbm(spw); _lg.fit(Xf_tr, yf_tr)

                oof_rf[val_idx]   = _rf.predict_proba(Xf_val)[:, 1]
                oof_xgb[val_idx]  = _xg.predict_proba(Xf_val)[:, 1]
                oof_lgbm[val_idx] = _lg.predict_proba(Xf_val)[:, 1]

            ## Train meta-learner on OOF predictions
            meta_X   = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
            meta_clf = make_meta()
            meta_clf.fit(meta_X, y_t)

            ## Retrain base models on full training data
            rf_full   = make_rf();       rf_full.fit(X_t, y_t)
            xgb_full  = make_xgb(spw);  xgb_full.fit(X_t, y_t)
            lgbm_full = make_lgbm(spw); lgbm_full.fit(X_t, y_t)

            model_bundle[target] = {
                "rf"       : rf_full,
                "xgb"      : xgb_full,
                "lgbm"     : lgbm_full,
                "meta"     : meta_clf,
                "spw"      : spw,
                "n_train"  : len(y_t),
                "pos_ratio": pos / len(y_t),
            }
            logger.info(f"  {target}: training complete.")

        ## Save test split for evaluation stage
        np.save(self.config.transformed_data_dir / "X_te.npy", X_te)
        np.save(self.config.transformed_data_dir / "y_te.npy", y_te)

        ## Save full model bundle
        joblib.dump(model_bundle, self.config.model_bundle_path, compress=3)
        size_mb = os.path.getsize(self.config.model_bundle_path) / 1e6
        logger.info(
            f"Model bundle saved → {self.config.model_bundle_path} ({size_mb:.1f} MB)"
        )
        return True