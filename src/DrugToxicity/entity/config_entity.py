## ENTITY
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    final_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path           # Input: artifacts/data_ingestion/tox21.csv
    features_path: Path       # Output: artifacts/data_transformation/X.npy
    labels_path: Path         # Output: artifacts/data_transformation/y.npy
    selector_path: Path       # Output: artifacts/data_transformation/selector.pkl
    scaler_path: Path  


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_data_dir: Path    # Input:  artifacts/data_transformation
    model_bundle_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    transformed_data_dir: Path   # Input: artifacts/data_transformation (X_te.npy, y_te.npy)
    model_bundle_path: Path      # Input: artifacts/model_trainer/tox21_model_bundle.pkl
    metric_file_name: Path       # Output: artifacts/model_evaluation/metrics.json
    roc_plot_path: Path          # Output: artifacts/model_evaluation/roc_curves.png
    pr_plot_path: Path     