"""
CodeCureAI — Tox21 Toxicity Predictor
Streamlit Cloud deployment
Models loaded from HuggingFace: GowthamD03/codecureai
"""

import os
import io
import warnings
import numpy as np
import joblib
import requests
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, Draw
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "CodeCureAI · Tox21 Predictor",
    page_icon   = "🧬",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS — dark lab aesthetic matching the Flask UI
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ──────────────────────────────────────────── */
:root {
  --bg0:#080c14; --bg1:#0e1421; --bg2:#131c2e; --bg3:#1a2540;
  --accent:#3ef0b4; --accent2:#4f8ef7;
  --danger:#f06c6c; --warn:#f5a623; --ok:#22c98e;
  --text0:#e8f0fe; --text1:#9aaac4; --text2:#5c6e8a;
  --border:#1f2e48;
  --mono:'Space Mono',monospace;
  --sans:'DM Sans',sans-serif;
}

/* ── Global overrides ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: #080c14 !important;
  font-family: 'DM Sans', sans-serif !important;
  color: #e8f0fe !important;
}
[data-testid="stAppViewContainer"] > .main { background: #080c14 !important; }
[data-testid="stHeader"]            { background: #080c14 !important; border-bottom: 1px solid #1f2e48; }
[data-testid="stSidebar"]           { background: #0e1421 !important; border-right: 1px solid #1f2e48; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0e1421; }
::-webkit-scrollbar-thumb { background: #1f2e48; border-radius: 3px; }

/* ── Inputs ───────────────────────────────────────────────── */
[data-testid="stTextInput"] > div > div > input {
  background: #131c2e !important;
  border: 1px solid #1f2e48 !important;
  border-radius: 8px !important;
  color: #e8f0fe !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.85rem !important;
  padding: 12px 16px !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
  border-color: #3ef0b4 !important;
  box-shadow: 0 0 0 3px rgba(62,240,180,.1) !important;
}
[data-testid="stTextInput"] label { color: #9aaac4 !important; font-family: 'Space Mono',monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; }

/* ── Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #3ef0b4, #2dd4a8) !important;
  color: #080c14 !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  padding: 0.55rem 1.6rem !important;
  transition: all .2s !important;
}
[data-testid="stButton"] > button:hover {
  box-shadow: 0 6px 24px rgba(62,240,180,.3) !important;
  transform: translateY(-1px) !important;
}

/* ── Selectbox ────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: #131c2e !important;
  border: 1px solid #1f2e48 !important;
  border-radius: 8px !important;
  color: #e8f0fe !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
}
[data-testid="stSelectbox"] label {
  color: #9aaac4 !important;
  font-family: 'Space Mono',monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #131c2e !important;
  border: 1px solid #1f2e48 !important;
  border-radius: 10px !important;
  padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #5c6e8a !important; font-size: 0.68rem !important; letter-spacing: 0.05em !important; }
[data-testid="stMetricValue"] { color: #e8f0fe !important; font-family: 'Space Mono',monospace !important; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
.stDataFrame iframe { border: 1px solid #1f2e48 !important; border-radius: 10px !important; }

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: #0e1421 !important;
  border: 1px solid #1f2e48 !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #9aaac4 !important; font-family: 'Space Mono',monospace !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; }

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: #1f2e48 !important; }

/* ── Info / success / error ───────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 3px !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #1f2e48 !important; gap: 4px; }
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  border: none !important;
  color: #5c6e8a !important;
  font-family: 'Space Mono',monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em !important;
  padding: 8px 16px !important;
  border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: #3ef0b4 !important;
  background: rgba(62,240,180,.07) !important;
  border-bottom: 2px solid #3ef0b4 !important;
}

/* ── Progress bar ─────────────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: #1a2540 !important; border-radius: 4px; }
[data-testid="stProgress"] > div > div > div { border-radius: 4px; }

/* ── Caption ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #5c6e8a !important; font-size: 0.72rem !important; }

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #3ef0b4 !important; }

/* ── Hide default streamlit branding ──────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
HF_BASE = "https://huggingface.co/GowthamD03/codecureai/resolve/main"
CACHE_DIR = "/tmp/tox21_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TARGET_COLS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

FP_COLORS = {
    "morgan" : "#4f8ef7",
    "maccs"  : "#22c98e",
    "rdkit"  : "#f5a623",
    "torsion": "#9b74e8",
    "desc"   : "#f06c6c",
}
FP_LABELS = {
    "morgan":"Morgan (ECFP4)", "maccs":"MACCS Keys",
    "rdkit":"RDKit FP", "torsion":"Torsion", "desc":"Descriptor"
}

QUICK_COMPOUNDS = {
    "Aspirin"   : "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine"  : "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Benzene"   : "c1ccccc1",
    "Ethanol"   : "CCO",
    "Ibuprofen" : "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "PCB-77"    : "Clc1ccc(cc1Cl)c2cc(Cl)c(Cl)cc2Cl",
}

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING (cached — runs once per session)
# ══════════════════════════════════════════════════════════════
def download_hf(filename):
    local = os.path.join(CACHE_DIR, filename)
    if os.path.exists(local):
        return local
    url = f"{HF_BASE}/{filename}"
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(local, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            f.write(chunk)
    return local

@st.cache_resource(show_spinner=False)
def load_all_models():
    files = {
        "bundle"  : "tox21_model_bundle.pkl",
        "selector": "selector.pkl",
        "scaler"  : "scaler.pkl",
        "X_te"    : "X_te.npy",
        "y_te"    : "y_te.npy",
    }
    loaded = {}
    prog = st.progress(0, text="Downloading models from HuggingFace…")
    for i, (key, fname) in enumerate(files.items()):
        prog.progress((i) / len(files), text=f"Loading {fname}…")
        path = download_hf(fname)
        if fname.endswith(".pkl"):
            loaded[key] = joblib.load(path)
        else:
            loaded[key] = np.load(path)
        prog.progress((i + 1) / len(files), text=f"✓ {fname}")

    prog.empty()

    raw_feat_names = (
        [f"morgan_{j}"  for j in range(2048)] +
        [f"maccs_{j}"   for j in range(167)]  +
        [f"rdkit_{j}"   for j in range(2048)] +
        [f"torsion_{j}" for j in range(256)]  +
        [d[0] for d in Descriptors.descList[:200]]
    )
    loaded["clean_feat"] = np.array(raw_feat_names)[loaded["selector"].get_support()]
    return loaded

# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
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

def prepare_features(smiles, models):
    feat = smiles_to_features(smiles).reshape(1, -1)
    feat = models["selector"].transform(feat)
    feat = models["scaler"].transform(feat)
    return np.clip(feat, -10, 10).astype(np.float32)

# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════
def predict_toxicity(smiles, models):
    feat = prepare_features(smiles, models)
    rows = []
    for target, mdls in models["bundle"].items():
        p_rf   = float(mdls["rf"].predict_proba(feat)[0, 1])
        p_xgb  = float(mdls["xgb"].predict_proba(feat)[0, 1])
        p_lgbm = float(mdls["lgbm"].predict_proba(feat)[0, 1])
        meta_f = np.array([[p_rf, p_xgb, p_lgbm]])
        prob   = float(mdls["meta"].predict_proba(meta_f)[0, 1])

        if prob >= 0.70:   risk = "🔴 High"
        elif prob >= 0.40: risk = "🟡 Medium"
        else:              risk = "🟢 Low"

        rows.append({
            "Target"      : target,
            "Toxic Prob"  : round(prob, 4),
            "Prediction"  : "Toxic" if prob >= 0.5 else "Non-toxic",
            "Risk"        : risk,
            "RF"          : round(p_rf, 4),
            "XGB"         : round(p_xgb, 4),
            "LGBM"        : round(p_lgbm, 4),
        })
    return sorted(rows, key=lambda x: x["Toxic Prob"], reverse=True)

# ══════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════
def mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(360, 260))
    return img

def feat_color(name):
    for k, v in FP_COLORS.items():
        if name.startswith(k):
            return v
    return FP_COLORS["desc"]

def plot_shap_bar(sv_bg, feat_names, target, n=20):
    mean_abs = np.abs(sv_bg).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n]
    names    = feat_names[top_idx]
    vals     = mean_abs[top_idx]
    colors   = [feat_color(nm) for nm in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    y_pos = np.arange(n)
    ax.barh(y_pos, vals[::-1], color=colors[::-1], height=0.68, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8, color="#c8d0e0")
    ax.set_xlabel("Mean |SHAP value|", color="#8892a4", fontsize=9)
    ax.set_title(f"Top {n} Features — {target}", color="#e2e8f0", fontsize=11, pad=10)
    ax.tick_params(colors="#8892a4")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3044")
    handles = [
        mpatches.Patch(color=v, label=FP_LABELS[k])
        for k, v in FP_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              facecolor="#1a2033", edgecolor="#2a3044", labelcolor="#c8d0e0")
    plt.tight_layout()
    return fig

def plot_shap_waterfall(sv_single, base_val, feat_single, feat_names, target, n=15):
    expl = shap.Explanation(
        values        = sv_single[0],
        base_values   = base_val,
        data          = feat_single[0],
        feature_names = feat_names,
    )
    plt.rcParams.update({"text.color": "#e2e8f0", "axes.labelcolor": "#8892a4"})
    shap.waterfall_plot(expl, max_display=n, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0e1117")
    for ax_ in fig.get_axes():
        ax_.set_facecolor("#131820")
        ax_.tick_params(colors="#8892a4")
        for sp in ax_.spines.values():
            sp.set_edgecolor("#2a3044")
    fig.suptitle(f"SHAP Waterfall — {target}", color="#e2e8f0", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black"})
    return fig

def styled_prob_html(prob):
    """Returns a small coloured probability pill."""
    if prob >= 0.70:   col, bg = "#f06c6c", "rgba(240,108,108,.12)"
    elif prob >= 0.40: col, bg = "#f5a623", "rgba(245,166,35,.1)"
    else:              col, bg = "#22c98e", "rgba(34,201,142,.1)"
    pct = prob * 100
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;height:6px;background:#1a2540;border-radius:3px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{col};border-radius:3px"></div></div>'
        f'<span style="font-family:Space Mono,monospace;font-size:.72rem;color:{col};min-width:42px">'
        f'{pct:.1f}%</span></div>'
    )

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="
  display:flex;align-items:center;justify-content:space-between;
  padding:24px 0 28px;
  border-bottom:1px solid #1f2e48;
  margin-bottom:36px;
">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="
      width:44px;height:44px;border-radius:10px;
      background:linear-gradient(135deg,#3ef0b4,#4f8ef7);
      display:flex;align-items:center;justify-content:center;
      font-size:22px;
    ">🧬</div>
    <div>
      <h1 style="font-family:'Space Mono',monospace;font-size:1.35rem;
                 letter-spacing:.05em;color:#e8f0fe;margin:0;line-height:1.2">
        CodeCureAI
      </h1>
      <p style="font-size:.75rem;color:#5c6e8a;margin:2px 0 0;letter-spacing:.03em">
        Tox21 Molecular Toxicity Predictor
      </p>
    </div>
  </div>
  <div style="
    font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:.1em;
    padding:5px 12px;border-radius:20px;
    border:1px solid rgba(62,240,180,.3);
    color:#3ef0b4;background:rgba(62,240,180,.06);
  ">XGB + RF + LightGBM Stack · mean ROC-AUC 0.85+</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "smiles"      not in st.session_state:
    st.session_state.smiles = ""
if "shap_result" not in st.session_state:
    st.session_state.shap_result = None
if "shap_target" not in st.session_state:
    st.session_state.shap_target = "SR-MMP"

# ══════════════════════════════════════════════════════════════
#  INPUT PANEL
# ══════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.15em;color:#3ef0b4;margin-bottom:10px">SMILES INPUT</p>', unsafe_allow_html=True)

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        smiles_input = st.text_input(
            label       = "SMILES",
            value       = st.session_state.smiles,
            placeholder = "Enter SMILES — e.g. CC(=O)Oc1ccccc1C(=O)O",
            label_visibility = "collapsed",
        )
    with col_btn:
        analyze_btn = st.button("ANALYZE →", use_container_width=True)

    # Quick compound chips
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:.62rem;color:#5c6e8a;margin:8px 0 6px">Quick examples:</p>', unsafe_allow_html=True)
    chip_cols = st.columns(len(QUICK_COMPOUNDS))
    for col, (name, smi) in zip(chip_cols, QUICK_COMPOUNDS.items()):
        with col:
            if st.button(name, key=f"chip_{name}",
                         help=smi,
                         use_container_width=True):
                st.session_state.smiles = smi
                st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════
active_smiles = smiles_input.strip() or st.session_state.smiles.strip()

if analyze_btn and active_smiles:
    st.session_state.smiles      = active_smiles
    st.session_state.predictions = None
    st.session_state.shap_result = None

    mol_check = Chem.MolFromSmiles(active_smiles)
    if mol_check is None:
        st.error("❌ Invalid SMILES string. Please check your input.")
    else:
        with st.spinner("Loading models and computing predictions…"):
            models = load_all_models()
        with st.spinner("Running XGB + RF + LightGBM stacking ensemble…"):
            preds = predict_toxicity(active_smiles, models)
        st.session_state.predictions = preds

# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if st.session_state.predictions:
    preds  = st.session_state.predictions
    smiles = st.session_state.smiles
    models = load_all_models()

    n_toxic  = sum(1 for p in preds if p["Prediction"] == "Toxic")
    n_safe   = len(preds) - n_toxic
    max_risk = max(p["Toxic Prob"] for p in preds)

    # ── Top row: molecule + summary stats ───────────────────
    col_mol, col_stats = st.columns([1, 2], gap="large")

    with col_mol:
        st.markdown("""
        <div style="background:#0e1421;border:1px solid #1f2e48;border-radius:12px;overflow:hidden">
          <div style="padding:14px 18px;border-bottom:1px solid #1f2e48">
            <span style="font-family:Space Mono,monospace;font-size:.7rem;letter-spacing:.1em;color:#9aaac4;text-transform:uppercase">Molecule</span>
          </div>
        """, unsafe_allow_html=True)

        mol_img = mol_image(smiles)
        if mol_img:
            st.image(mol_img, use_container_width=True)

        st.markdown(f"""
          <div style="padding:0 16px 16px">
            <div style="font-family:Space Mono,monospace;font-size:.68rem;color:#5c6e8a;
                        word-break:break-all;line-height:1.6;
                        background:#131c2e;border-radius:6px;padding:8px 10px;margin-top:10px">
              {smiles}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Toxic Endpoints",  n_toxic,  delta=None)
        with mc2:
            st.metric("Safe Endpoints",   n_safe,   delta=None)
        with mc3:
            st.metric("Total Targets",    len(preds))
        with mc4:
            st.metric("Max Risk Score",   f"{max_risk:.1%}")

        # Predictions table - using st.dataframe instead of raw HTML
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.12em;color:#3ef0b4;margin:18px 0 10px">PREDICTIONS — 12 TOX21 ENDPOINTS</p>', unsafe_allow_html=True)
        
        # Create DataFrame for predictions
        df_predictions = pd.DataFrame(preds)
        
        # Format the probability column with color coding
        def style_probability(val):
            if val >= 0.70:
                color = '#f06c6c'
            elif val >= 0.40:
                color = '#f5a623'
            else:
                color = '#22c98e'
            return f'color: {color}; font-weight: bold'
        
        def style_prediction(val):
            color = '#f06c6c' if val == 'Toxic' else '#22c98e'
            return f'color: {color}; font-weight: bold'
        
        # Apply styling to the dataframe
        styled_df = df_predictions.style.applymap(style_probability, subset=['Toxic Prob'])
        styled_df = styled_df.applymap(style_prediction, subset=['Prediction'])
        
        # Display the dataframe
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── SHAP Section ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 20px">
      <div style="flex:1;height:1px;background:#1f2e48"></div>
      <span style="font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.15em;
                   color:#5c6e8a;text-transform:uppercase;white-space:nowrap">SHAP Explainability</span>
      <div style="flex:1;height:1px;background:#1f2e48"></div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP selector and legend in two columns
    shap_col1, shap_col2 = st.columns([1, 3])
    with shap_col1:
        shap_target = st.selectbox(
            "SELECT TARGET FOR SHAP ANALYSIS",
            TARGET_COLS,
            index=TARGET_COLS.index(st.session_state.shap_target) if st.session_state.shap_target in TARGET_COLS else 0,
            key="shap_select_main"
        )
        st.session_state.shap_target = shap_target
        shap_btn = st.button("⚡ Compute SHAP", key="shap_btn_main", use_container_width=True)

        st.markdown("""
        <div style="background:#131c2e;border:1px solid #1f2e48;border-radius:8px;padding:14px;margin-top:12px">
          <p style="font-family:Space Mono,monospace;font-size:.6rem;letter-spacing:.08em;
                    color:#5c6e8a;text-transform:uppercase;margin-bottom:10px">Feature Types</p>
          <div style="display:flex;flex-direction:column;gap:7px">
            <span style="font-size:.7rem;color:#4f8ef7;font-family:Space Mono,monospace">⬛ Morgan (ECFP4)</span>
            <span style="font-size:.7rem;color:#22c98e;font-family:Space Mono,monospace">⬛ MACCS Keys</span>
            <span style="font-size:.7rem;color:#f5a623;font-family:Space Mono,monospace">⬛ RDKit FP</span>
            <span style="font-size:.7rem;color:#9b74e8;font-family:Space Mono,monospace">⬛ Torsion</span>
            <span style="font-size:.7rem;color:#f06c6c;font-family:Space Mono,monospace">⬛ Descriptor</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with shap_col2:
        if shap_btn:
            st.session_state.shap_result = None
            with st.spinner(f"Computing TreeExplainer SHAP values for {shap_target}…"):
                bundle = models["bundle"]
                if shap_target in bundle:
                    feat     = prepare_features(smiles, models)
                    lgbm_mdl = bundle[shap_target]["lgbm"]
                    t_idx    = TARGET_COLS.index(shap_target)
                    mask     = models["y_te"][:, t_idx] != -1
                    X_bg     = models["X_te"][mask][:150]

                    explainer   = shap.TreeExplainer(lgbm_mdl)
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

                    clean_feat = models["clean_feat"]
                    mean_abs   = np.abs(sv_bg).mean(axis=0)
                    top_idx    = np.argsort(mean_abs)[::-1][:10]

                    top_feats = [{
                        "Feature"       : clean_feat[i],
                        "Type"          : next((FP_LABELS[k] for k in FP_COLORS if clean_feat[i].startswith(k)), "Descriptor"),
                        "Mean |SHAP|"   : round(float(mean_abs[i]), 5),
                        "SHAP (this mol)": round(float(sv_single[0][i]), 5),
                    } for i in top_idx]

                    st.session_state.shap_result = {
                        "sv_bg"      : sv_bg,
                        "sv_single"  : sv_single,
                        "base_val"   : base_val,
                        "feat"       : feat,
                        "clean_feat" : clean_feat,
                        "top_feats"  : top_feats,
                        "target"     : shap_target,
                    }

        if st.session_state.shap_result:
            sr = st.session_state.shap_result

            tab1, tab2, tab3 = st.tabs([
                "📊  Feature Importance (Mean |SHAP|)",
                "🌊  Waterfall (This Molecule)",
                "📋  Top 10 Feature Table",
            ])

            with tab1:
                fig_bar = plot_shap_bar(sr["sv_bg"], sr["clean_feat"], sr["target"])
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

            with tab2:
                fig_wf = plot_shap_waterfall(
                    sr["sv_single"], sr["base_val"],
                    sr["feat"], sr["clean_feat"], sr["target"]
                )
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

            with tab3:
                df_feats = pd.DataFrame(sr["top_feats"])

                def color_shap(val):
                    if isinstance(val, float):
                        c = "#4f8ef7" if val >= 0 else "#f06c6c"
                        return f"color: {c}; font-family: Space Mono, monospace; font-size: 0.75rem"
                    return ""

                def color_type(val):
                    rev = {v: k for k, v in FP_LABELS.items()}
                    key = rev.get(val, "desc")
                    c = FP_COLORS.get(key, "#aaa")
                    return f"color: {c}; font-family: Space Mono, monospace; font-size: 0.72rem"

                styled = (
                    df_feats.style
                    .applymap(color_shap, subset=["Mean |SHAP|", "SHAP (this mol)"])
                    .applymap(color_type, subset=["Type"])
                    .set_properties(**{
                        "background-color": "#0e1421",
                        "color"           : "#9aaac4",
                        "border"          : "1px solid #1f2e48",
                        "font-size"       : "0.78rem",
                    })
                    .set_table_styles([{
                        "selector": "th",
                        "props": [
                            ("background-color", "#131c2e"),
                            ("color", "#5c6e8a"),
                            ("font-family", "Space Mono, monospace"),
                            ("font-size", "0.62rem"),
                            ("letter-spacing", "0.1em"),
                            ("text-transform", "uppercase"),
                            ("border", "1px solid #1f2e48"),
                        ]
                    }])
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            if shap_btn:
                st.info(f"SHAP analysis requested for {shap_target}. Results will appear here.")
            else:
                st.markdown("""
                <div style="
                  display:flex;flex-direction:column;align-items:center;justify-content:center;
                  padding:60px 40px;color:#5c6e8a;text-align:center;
                  background:#0e1421;border:1px solid #1f2e48;border-radius:12px;
                ">
                  <div style="font-size:2.5rem;margin-bottom:12px;opacity:.4">📊</div>
                  <p style="font-size:.85rem;line-height:1.7;margin:0">
                    Select a target endpoint and click<br>
                    <strong style="color:#3ef0b4;font-family:Space Mono,monospace">⚡ Compute SHAP</strong><br>
                    to see feature importance explanations.
                  </p>
                </div>
                """, unsafe_allow_html=True)

else:
    # ── Landing empty state ──────────────────────────────────
    if not (analyze_btn and active_smiles):
        st.markdown("""
        <div style="
          display:flex;flex-direction:column;align-items:center;justify-content:center;
          padding:100px 40px;color:#5c6e8a;text-align:center;
        ">
          <div style="font-size:4rem;margin-bottom:20px;opacity:.35">🔬</div>
          <h2 style="font-family:Space Mono,monospace;font-size:1rem;color:#9aaac4;
                     letter-spacing:.08em;margin-bottom:14px">
            Enter a SMILES string above to get started
          </h2>
          <p style="font-size:.82rem;line-height:1.8;max-width:520px;margin:0">
            Predict toxicity across 12 Tox21 biological assay endpoints using a
            stacking ensemble of XGBoost, RandomForest, and LightGBM — with
            SHAP explainability powered by TreeExplainer.
          </p>
          <div style="
            display:flex;gap:24px;margin-top:36px;flex-wrap:wrap;justify-content:center
          ">
            <div style="text-align:center">
              <div style="font-family:Space Mono,monospace;font-size:1.4rem;color:#3ef0b4;font-weight:700">12</div>
              <div style="font-size:.7rem;color:#5c6e8a;margin-top:4px">Tox21 Endpoints</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:Space Mono,monospace;font-size:1.4rem;color:#4f8ef7;font-weight:700">3</div>
              <div style="font-size:.7rem;color:#5c6e8a;margin-top:4px">Base Models</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:Space Mono,monospace;font-size:1.4rem;color:#f5a623;font-weight:700">0.85+</div>
              <div style="font-size:.7rem;color:#5c6e8a;margin-top:4px">Mean ROC-AUC</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:Space Mono,monospace;font-size:1.4rem;color:#9b74e8;font-weight:700">SHAP</div>
              <div style="font-size:.7rem;color:#5c6e8a;margin-top:4px">Explainability</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)