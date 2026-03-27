"""
Tox21 Toxicity Prediction — Flask Backend
Downloads models from HuggingFace on first run, caches locally.
"""

import os
import io
import base64
import warnings
import numpy as np
import joblib
import requests
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── RDKit ────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, Draw
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── HuggingFace config ───────────────────────────────────────
HF_REPO   = "GowthamD03/codecureai"
HF_BASE   = f"https://huggingface.co/{HF_REPO}/resolve/main"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TARGET_COLS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

def download_file(filename):
    local = os.path.join(CACHE_DIR, filename)
    if os.path.exists(local):
        print(f"  [cache] {filename}")
        return local
    url = f"{HF_BASE}/{filename}"
    print(f"  [download] {filename} …")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(local, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  [done] {filename}")
    return local

# ── Global model state ───────────────────────────────────────
MODEL_BUNDLE = None
SELECTOR     = None
SCALER       = None
X_TE         = None
Y_TE         = None
CLEAN_FEAT   = None

def load_models():
    global MODEL_BUNDLE, SELECTOR, SCALER, X_TE, Y_TE, CLEAN_FEAT
    if MODEL_BUNDLE is not None:
        return

    print("Loading models from HuggingFace …")
    MODEL_BUNDLE = joblib.load(download_file("tox21_model_bundle.pkl"))
    SELECTOR     = joblib.load(download_file("selector.pkl"))
    SCALER       = joblib.load(download_file("scaler.pkl"))
    X_TE         = np.load(download_file("X_te.npy"))
    Y_TE         = np.load(download_file("y_te.npy"))

    # Build clean feature names
    raw_feat_names = (
        [f"morgan_{j}"  for j in range(2048)] +
        [f"maccs_{j}"   for j in range(167)]  +
        [f"rdkit_{j}"   for j in range(2048)] +
        [f"torsion_{j}" for j in range(256)]  +
        [d[0] for d in Descriptors.descList[:200]]
    )
    CLEAN_FEAT = np.array(raw_feat_names)[SELECTOR.get_support()]
    print(f"  Models loaded. Clean features: {len(CLEAN_FEAT)}")

# ── Feature engineering (mirrors training code) ──────────────
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(4719, dtype=np.float32)
    morgan  = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
    maccs   = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    rdkitfp = np.array(Chem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)
    torsion = np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=256), dtype=np.float32)
    desc = []
    for name, _ in Descriptors.descList[:200]:
        try:
            val = float(getattr(Descriptors, name)(mol))
            if not np.isfinite(val): val = 0.0
        except:
            val = 0.0
        desc.append(val)
    feat = np.concatenate([morgan, maccs, rdkitfp, torsion, np.array(desc, dtype=np.float32)])
    return np.clip(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)

def prepare_features(smiles):
    feat = smiles_to_features(smiles).reshape(1, -1)
    feat = SELECTOR.transform(feat)
    feat = SCALER.transform(feat)
    return np.clip(feat, -10, 10).astype(np.float32)

# ── Molecule image ───────────────────────────────────────────
def mol_to_b64(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(300, 220))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

# ── SHAP bar chart ───────────────────────────────────────────
FP_COLORS = {
    "morgan" : "#4f8ef7",
    "maccs"  : "#22c98e",
    "rdkit"  : "#f5a623",
    "torsion": "#9b74e8",
    "desc"   : "#f06c6c",
}

def feat_color(name):
    for k, v in FP_COLORS.items():
        if name.startswith(k):
            return v
    return FP_COLORS["desc"]

def shap_bar_b64(shap_vals, feat_names, target, n=20):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n]
    names    = feat_names[top_idx]
    vals     = mean_abs[top_idx]
    colors   = [feat_color(nm) for nm in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    y_pos = np.arange(n)
    ax.barh(y_pos, vals[::-1], color=colors[::-1], height=0.7, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8, color="#c8d0e0")
    ax.set_xlabel("Mean |SHAP value|", color="#8892a4", fontsize=9)
    ax.set_title(f"Top {n} Features — {target}", color="#e2e8f0", fontsize=11, pad=10)
    ax.tick_params(colors="#8892a4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3044")

    legend_handles = [
        mpatches.Patch(color=v, label={"morgan":"Morgan (ECFP4)","maccs":"MACCS Keys",
                                        "rdkit":"RDKit FP","torsion":"Torsion","desc":"Descriptor"}[k])
        for k, v in FP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              facecolor="#1a2033", edgecolor="#2a3044", labelcolor="#c8d0e0")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def shap_waterfall_b64(sv_single, base_val, feat_single, feat_names, target, n=15):
    expl = shap.Explanation(
        values       = sv_single[0],
        base_values  = base_val,
        data         = feat_single[0],
        feature_names= feat_names,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0e1117")
    plt.rcParams.update({"text.color": "#e2e8f0", "axes.labelcolor": "#8892a4"})
    shap.waterfall_plot(expl, max_display=n, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0e1117")
    for ax_ in fig.get_axes():
        ax_.set_facecolor("#131820")
        ax_.tick_params(colors="#8892a4")
        for spine in ax_.spines.values():
            spine.set_edgecolor("#2a3044")
    plt.title(f"SHAP Waterfall — {target}", color="#e2e8f0", fontsize=11, pad=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black"})
    return base64.b64encode(buf.getvalue()).decode()

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(force=True)
    smiles = data.get("smiles", "").strip()
    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400

    load_models()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400

    feat = prepare_features(smiles)

    predictions = []
    for target, models in MODEL_BUNDLE.items():
        p_rf   = float(models["rf"].predict_proba(feat)[0, 1])
        p_xgb  = float(models["xgb"].predict_proba(feat)[0, 1])
        p_lgbm = float(models["lgbm"].predict_proba(feat)[0, 1])
        meta_f = np.array([[p_rf, p_xgb, p_lgbm]])
        prob   = float(models["meta"].predict_proba(meta_f)[0, 1])

        if prob >= 0.70:
            risk, conf = "high",   "High"
        elif prob >= 0.40:
            risk, conf = "medium", "Medium"
        else:
            risk, conf = "low",    "Low"

        predictions.append({
            "target"    : target,
            "toxic_prob": round(prob, 4),
            "prediction": "Toxic" if prob >= 0.5 else "Non-toxic",
            "confidence": conf,
            "risk"      : risk,
            "base_probs": {"rf": round(p_rf,4), "xgb": round(p_xgb,4), "lgbm": round(p_lgbm,4)},
        })

    predictions.sort(key=lambda x: x["toxic_prob"], reverse=True)

    mol_img = mol_to_b64(smiles)
    n_toxic = sum(1 for p in predictions if p["prediction"] == "Toxic")

    return jsonify({
        "smiles"      : smiles,
        "mol_image"   : mol_img,
        "predictions" : predictions,
        "n_toxic"     : n_toxic,
        "n_targets"   : len(predictions),
    })

@app.route("/shap", methods=["POST"])
def compute_shap():
    data   = request.get_json(force=True)
    smiles = data.get("smiles", "").strip()
    target = data.get("target", "SR-MMP")

    load_models()

    if target not in MODEL_BUNDLE:
        return jsonify({"error": f"Target {target} not found"}), 400

    feat = prepare_features(smiles)

    lgbm_model = MODEL_BUNDLE[target]["lgbm"]

    # Background — use 150 samples from X_te for the target
    t_idx  = TARGET_COLS.index(target) if target in TARGET_COLS else 0
    mask   = Y_TE[:, t_idx] != -1
    X_bg   = X_TE[mask][:150]

    explainer   = shap.TreeExplainer(lgbm_model)
    sv_bg       = explainer.shap_values(X_bg)
    sv_single   = explainer.shap_values(feat)

    if isinstance(sv_bg, list):
        sv_bg    = sv_bg[1]
        sv_single= sv_single[1]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])
    else:
        base_val = float(base_val)

    bar_img       = shap_bar_b64(sv_bg, CLEAN_FEAT, target)
    waterfall_img = shap_waterfall_b64(sv_single, base_val, feat, CLEAN_FEAT, target)

    # Top features table
    mean_abs = np.abs(sv_bg).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:10]
    top_feats = [
        {"name": CLEAN_FEAT[i], "importance": round(float(mean_abs[i]), 5),
         "shap_single": round(float(sv_single[0][i]), 5),
         "type": next((k for k in FP_COLORS if CLEAN_FEAT[i].startswith(k)), "desc")}
        for i in top_idx
    ]

    return jsonify({
        "target"       : target,
        "bar_chart"    : bar_img,
        "waterfall"    : waterfall_img,
        "top_features" : top_feats,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)