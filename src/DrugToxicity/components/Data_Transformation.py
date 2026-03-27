import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from src.DrugToxicity.utils.common import logger
from src.DrugToxicity.entity.config_entity import DataTransformationConfig

## RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

def smiles_to_features(smiles: str) -> np.ndarray:
    """
    Convert SMILES → 4719-dimensional feature vector:
      1. Morgan ECFP4     (2048 bits)
      2. MACCS keys       ( 167 bits)
      3. RDKit path FP    (2048 bits)
      4. Topological tors ( 256 bits)
      5. RDKit descriptors( 200 floats)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(4719, dtype=np.float32)

    morgan  = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
    maccs   = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    rdkitfp = np.array(Chem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)
    torsion = np.array(
        rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=256),
        dtype=np.float32
    )

    desc_vals = []
    for name, _ in Descriptors.descList[:200]:
        try:
            val = float(getattr(Descriptors, name)(mol))
            if not np.isfinite(val):
                val = 0.0
        except Exception:
            val = 0.0
        desc_vals.append(val)

    feat = np.concatenate([morgan, maccs, rdkitfp, torsion, np.array(desc_vals, dtype=np.float32)])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(feat, -1e4, 1e4).astype(np.float32)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
        """
        Skips entirely if X.npy already exists in artifacts/data_transformation.
        Otherwise:
          1. Builds 4719-dim molecular fingerprint + descriptor features
          2. Removes near-zero variance features (VarianceThreshold 0.01)
          3. Scales with RobustScaler (sparse-safe, with_centering=False)
          4. Saves X.npy, y.npy, selector.pkl, scaler.pkl
        """
        if self.config.features_path.exists():
            logger.info(
                f"Transformed features already exist at {self.config.features_path} "
                "— skipping transformation."
            )
            return False

        logger.info("Starting data transformation ...")
        os.makedirs(self.config.root_dir, exist_ok=True)

        ## Load data
        df = pd.read_csv(self.config.data_path)
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} compounds after dropping null SMILES.")

        ## Build raw feature matrix
        logger.info("Engineering features (Morgan + MACCS + RDKit FP + Torsion + Descriptors) ...")
        X_raw = np.vstack(df["smiles"].apply(smiles_to_features).values)
        logger.info(f"Raw feature matrix: {X_raw.shape}")

        ## Remove near-zero variance features
        selector = VarianceThreshold(threshold=0.01)
        X_sel = selector.fit_transform(X_raw)
        logger.info(f"After VarianceThreshold(0.01): {X_sel.shape}")

        ## Scale — RobustScaler, sparse-safe (with_centering=False)
        scaler = RobustScaler(with_centering=False)
        X_scaled = scaler.fit_transform(X_sel)
        X_scaled = np.clip(X_scaled, -10, 10).astype(np.float32)
        logger.info(f"Scaled & clipped. Final shape: {X_scaled.shape}")

        ## Build target matrix — NaN → -1 sentinel (masked loss pattern)
        y = np.where(
            np.isnan(df[TARGET_COLS].values.astype(float)),
            -1,
            df[TARGET_COLS].values.astype(float)
        )
        logger.info(f"Target matrix shape: {y.shape}")

        ## Save outputs
        np.save(self.config.features_path, X_scaled)
        np.save(self.config.labels_path,   y)
        joblib.dump(selector, self.config.selector_path)
        joblib.dump(scaler,   self.config.scaler_path)

        logger.info(
            f"Saved: X.npy{X_scaled.shape}, y.npy{y.shape}, "
            f"selector.pkl, scaler.pkl → {self.config.root_dir}"
        )
        return True