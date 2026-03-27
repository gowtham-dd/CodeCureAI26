import pandas as pd
from pathlib import Path
from src.DrugToxicity.utils.common import logger
from src.DrugToxicity.entity.config_entity import DataValidationConfig

## Expected Tox21 schema — columns and their pandas dtypes
TOX21_SCHEMA = {
    "smiles"       : "object",
    "mol_id"       : "object",
    "NR-AR"        : "float64",
    "NR-AR-LBD"    : "float64",
    "NR-AhR"       : "float64",
    "NR-Aromatase" : "float64",
    "NR-ER"        : "float64",
    "NR-ER-LBD"    : "float64",
    "NR-PPAR-gamma": "float64",
    "SR-ARE"       : "float64",
    "SR-ATAD5"     : "float64",
    "SR-HSE"       : "float64",
    "SR-MMP"       : "float64",
    "SR-p53"       : "float64",
}


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self) -> bool:
        """
        Validates the Tox21 CSV:
          1. Skips if STATUS_FILE already exists
          2. All required columns present
          3. Column dtypes match schema
          4. smiles column has at least some non-null values
          5. At least one target column has labelled rows
          6. Minimum row count sanity check
        Writes result to STATUS_FILE.
        Returns True if valid, raises ValueError if not.
        """
        status_path = Path(self.config.STATUS_FILE)

        ## Skip if already validated
        if status_path.exists():
            logger.info(
                f"Validation status file already exists at {status_path} — skipping validation."
            )
            with open(status_path) as f:
                return "True" in f.read()

        validation_status = True
        issues = []

        try:
            df = pd.read_csv(self.config.data_path)

            ## 1. Required columns present
            missing = set(TOX21_SCHEMA.keys()) - set(df.columns)
            if missing:
                issues.append(f"Missing columns: {missing}")
                validation_status = False

            ## 2. Data types
            for col, expected_type in TOX21_SCHEMA.items():
                if col not in df.columns:
                    continue
                actual = str(df[col].dtype)
                if expected_type == "float64" and "float" not in actual:
                    issues.append(f"Type mismatch '{col}': expected float64, got {actual}")
                    validation_status = False
                if expected_type == "object" and actual != "object":
                    issues.append(f"Type mismatch '{col}': expected object, got {actual}")
                    validation_status = False

            ## 3. smiles not all null
            if "smiles" in df.columns and df["smiles"].isna().all():
                issues.append("All SMILES values are null")
                validation_status = False

            ## 4. At least one target has labels
            target_cols = [c for c in TOX21_SCHEMA if c not in ("smiles", "mol_id")]
            present = [c for c in target_cols if c in df.columns]
            if present and df[present].notna().sum().max() == 0:
                issues.append("All target columns are entirely null")
                validation_status = False

            ## 5. Minimum row count
            if len(df) < 100:
                issues.append(f"Dataset too small: only {len(df)} rows")
                validation_status = False

            logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            issues.append(f"Exception during validation: {e}")
            validation_status = False

        ## Write status file
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w") as f:
            f.write(f"Validation status: {validation_status}\n")
            for issue in issues:
                f.write(f"  - {issue}\n")

        if validation_status:
            logger.info("Data validation PASSED.")
        else:
            logger.error(f"Data validation FAILED. Issues: {issues}")

        return validation_status