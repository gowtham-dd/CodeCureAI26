import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.DrugToxicity.entity.config_entity import ModelEvaluationConfig


from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score,
    roc_curve, precision_recall_curve
)
from src.DrugToxicity.utils.common import logger, save_json

TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        """
        Skips if metrics.json already exists.
        Otherwise evaluates the stacking ensemble on the held-out test set:
          - Loads X_te.npy, y_te.npy saved by model training stage
          - Runs RF → XGB → LGBM → meta LR stacking prediction per target
          - Computes ROC-AUC, PR-AUC, Accuracy, F1 per target
          - Saves metrics.json, roc_curves.png, pr_curves.png
        Returns True if evaluated, False if skipped.
        """
        if self.config.metric_file_name.exists():
            logger.info(
                f"Metrics already exist at {self.config.metric_file_name} "
                "— skipping evaluation."
            )
            return False

        os.makedirs(self.config.root_dir, exist_ok=True)

        ## Load held-out test split
        X_te = np.load(self.config.transformed_data_dir / "X_te.npy")
        y_te = np.load(self.config.transformed_data_dir / "y_te.npy")
        logger.info(f"Test set: X{X_te.shape} y{y_te.shape}")

        ## Load model bundle
        bundle = joblib.load(self.config.model_bundle_path)
        logger.info(f"Model bundle loaded from {self.config.model_bundle_path}")

        metrics  = {}
        roc_data = {}
        pr_data  = {}

        for i, target in enumerate(TARGET_COLS):
            if target not in bundle:
                logger.warning(f"  {target}: not in bundle — skipping.")
                continue

            mask = y_te[:, i] != -1
            X_t  = X_te[mask]
            y_t  = y_te[mask, i]

            if len(np.unique(y_t)) < 2 or len(y_t) == 0:
                logger.warning(f"  {target}: insufficient test labels — skipping.")
                continue

            models = bundle[target]

            ## Stacking prediction
            p_rf   = models["rf"].predict_proba(X_t)[:, 1]
            p_xgb  = models["xgb"].predict_proba(X_t)[:, 1]
            p_lgbm = models["lgbm"].predict_proba(X_t)[:, 1]
            meta_X = np.column_stack([p_rf, p_xgb, p_lgbm])
            prob   = models["meta"].predict_proba(meta_X)[:, 1]
            pred   = (prob >= 0.5).astype(int)

            roc = roc_auc_score(y_t, prob)
            pr  = average_precision_score(y_t, prob)
            acc = accuracy_score(y_t, pred)
            f1  = f1_score(y_t, pred, zero_division=0)

            metrics[target] = {
                "roc_auc" : round(roc, 4),
                "pr_auc"  : round(pr,  4),
                "accuracy": round(acc, 4),
                "f1_score": round(f1,  4),
                "n_test"  : int(mask.sum()),
                "n_pos"   : int(y_t.sum()),
            }

            ## Store curve data for plots
            fpr, tpr, _ = roc_curve(y_t, prob)
            pre, rec, _ = precision_recall_curve(y_t, prob)
            roc_data[target] = (fpr, tpr, roc)
            pr_data[target]  = (rec, pre, pr)

            logger.info(
                f"  {target}: ROC={roc:.4f}  PR={pr:.4f}  "
                f"ACC={acc:.4f}  F1={f1:.4f}"
            )

        ## Summary stats
        valid = list(metrics.values())
        if valid:
            metrics["__summary__"] = {
                "mean_roc_auc" : round(np.mean([v["roc_auc"]  for v in valid]), 4),
                "mean_pr_auc"  : round(np.mean([v["pr_auc"]   for v in valid]), 4),
                "mean_accuracy": round(np.mean([v["accuracy"] for v in valid]), 4),
                "mean_f1"      : round(np.mean([v["f1_score"] for v in valid]), 4),
                "n_targets"    : len(valid),
            }
            s = metrics["__summary__"]
            logger.info(
                f"\nSUMMARY — Mean ROC-AUC: {s['mean_roc_auc']}  "
                f"Mean PR-AUC: {s['mean_pr_auc']}  "
                f"Mean Acc: {s['mean_accuracy']}"
            )

        ## Save metrics.json
        save_json(self.config.metric_file_name, metrics)

        ## Save ROC curves plot
        self._plot_roc(roc_data)

        ## Save PR curves plot
        self._plot_pr(pr_data)

        return True

    def _plot_roc(self, roc_data: dict):
        n    = len(roc_data)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten()
        colors = plt.cm.tab10.colors

        for idx, (target, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
            ax = axes[idx]
            ax.plot(fpr, tpr, color=colors[idx % 10], lw=2, label=f"AUC={auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_title(target, fontsize=9)
            ax.set_xlabel("FPR", fontsize=8)
            ax.set_ylabel("TPR", fontsize=8)
            ax.legend(fontsize=8)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

        for j in range(len(roc_data), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "ROC Curves — Tox21 Stacking Ensemble (XGB + RF + LGBM)",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.savefig(self.config.roc_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curves saved → {self.config.roc_plot_path}")

    def _plot_pr(self, pr_data: dict):
        n    = len(pr_data)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten()
        colors = plt.cm.tab10.colors

        for idx, (target, (rec, pre, ap)) in enumerate(pr_data.items()):
            ax = axes[idx]
            ax.plot(rec, pre, color=colors[idx % 10], lw=2, label=f"AP={ap:.3f}")
            ax.set_title(target, fontsize=9)
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.legend(fontsize=8)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

        for j in range(len(pr_data), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Precision-Recall Curves — Tox21 Stacking Ensemble",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.savefig(self.config.pr_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"PR curves saved → {self.config.pr_plot_path}")