# -*- coding: utf-8 -*-
# Streamlit NMR Workbench — MNova .csv | Layout patterned after your HPLC app
import io
import os
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: SciPy for normality tests (Shapiro / D'Agostino / Anderson)
try:
    from scipy import stats as sstats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ------------------ Robust import of your NMR toolbox ------------------
NMR_AVAILABLE = True
_NMR_IMPORT_ERR = None
try:
    import importlib.util, pathlib
    MOD_PATH = pathlib.Path(__file__).parent / "data_processing_NMR.py"
    spec = importlib.util.spec_from_file_location("nmrtools", MOD_PATH)
    nmr = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(nmr)
except Exception as e:
    NMR_AVAILABLE = False
    _NMR_IMPORT_ERR = e

# ------------------ Optional deps for alignment & modeling -------------
try:
    from pyicoshift import icoshift  # noqa: F401
    HAVE_PYICOSHIFT = True
except Exception:
    HAVE_PYICOSHIFT = False

SKLEARN_AVAILABLE = True
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import r2_score
except Exception as _e:
    SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e

PYARROW_AVAILABLE = True
try:
    import pyarrow  # noqa: F401
except Exception:
    PYARROW_AVAILABLE = False

# ------------------ Page & header --------------------------------------
st.set_page_config(page_title="pyMETAflow NMR (MNova .csv)", layout="wide")
st.title("pyMETAflow NMR")
st.caption("Parse MNova .csv, preprocess, mask ppm, reference to any ppm, align (iCOshift/PAFFT/RAFFT), visualize, STOCSY, and model (PCA/PLS-DA).")


# Optional logos (won't crash if missing)
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["pyMETAflow_NMR.png", "LAABio.png"]:
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

# Helpful links
st.markdown(
    """
Upload **MNova .csv** exports (two columns: *Chemical Shift (ppm)* and *Intensity*).  
The app merges on the ppm axis, trims/masks ppm, lets you **reference to any target peak** (e.g., TMS 0.00, DMSO-d6 2.50),
align, normalize/scale, run PCA/PLS-DA, STOCSY, and export.
"""
)

st.sidebar.markdown("---")

# ------------------ Sidebar: uploads & preferences ---------------------
st.sidebar.header("1) Upload NMR files (.csv)")
uploads = st.sidebar.file_uploader(
    "MNova .csv exports (multiple)", type=["csv"], accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.header("2) Upload metadata (.csv)")
meta_file = st.sidebar.file_uploader("Metadata CSV (semicolon ';' expected)", type=["csv", "txt"])

meta_df = None
if meta_file is not None:
    # Always try semicolon first; if that still yields 1 column, try inference.
    for enc in ("utf-8-sig", "latin1"):
        try:
            meta_file.seek(0)
            df_try = pd.read_csv(meta_file, sep=";", engine="python", encoding=enc)
            if df_try.shape[1] > 1:
                meta_df = df_try
                break
        except Exception:
            pass
    if meta_df is None:
        # fallback: let pandas sniff the delimiter
        try:
            meta_file.seek(0)
            meta_df = pd.read_csv(meta_file, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            st.error(f"Failed to read metadata: {e}")

    if meta_df is not None and meta_df.shape[1] == 1:
        st.warning("Metadata parsed as a single column. This usually means the delimiter wasn't detected. "
                   "Confirm the file is semicolon-delimited or try saving as UTF-8 (CSV;).")

# ------------------ Utilities -----------------------------------------
def show_df(df: pd.DataFrame):
    """Display a DataFrame safely even without pyarrow."""
    use_static = st.session_state.get("use_static_table", True)
    if (not use_static) and PYARROW_AVAILABLE:
        st.dataframe(df, use_container_width=True)
    else:
        try:
            st.markdown(df.to_html(index=False), unsafe_allow_html=True)
        except Exception:
            st.text(df.to_string(index=False))

def parse_mask_regions(mask_str: str) -> List[Tuple[float, float]]:
    if not mask_str.strip():
        return []
    out = []
    for chunk in mask_str.split(";"):
        if not chunk.strip():
            continue
        try:
            a, b = chunk.split(":")
            out.append((float(a.strip()), float(b.strip())))
        except Exception:
            st.warning(f"Could not parse region '{chunk}'. Use 'start:end; start:end'.")
    return out

def outer_merge_on_ppm(dfs: List[pd.DataFrame], axis_col="Chemical Shift (ppm)") -> pd.DataFrame:
    base = dfs[0].copy()
    for df in dfs[1:]:
        base = base.merge(df, on=axis_col, how="outer")
    return base.sort_values(axis_col).reset_index(drop=True)

def interpolate_common(df: pd.DataFrame, axis_col="Chemical Shift (ppm)") -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c == axis_col:
            continue
        y = pd.to_numeric(df[c], errors="coerce")
        y = y.interpolate("linear").bfill().ffill()
        df[c] = y
    return df

def matrix_to_XY(df: pd.DataFrame, sample_order: List[str], axis_col="Chemical Shift (ppm)") -> tuple:
    """Return X (samples x points), features list (ppm strings)."""
    X = np.vstack([df[s].astype(float).values for s in sample_order])
    ppm = [f"{p:.5f}" for p in df[axis_col].values]
    return X, ppm

def coalesce_df(*candidates):
    """Return the first argument that is a non-empty pandas DataFrame; else None."""
    for x in candidates:
        if isinstance(x, pd.DataFrame) and not x.empty:
            return x
    for x in candidates:
        if isinstance(x, pd.DataFrame):
            return x
    return None


# ------------------ Import step (MNova .csv) ---------------------------
combined = None
report_rows = []

if not NMR_AVAILABLE:
    st.error("Could not import data_processing_NMR.py.")
    st.code(str(_NMR_IMPORT_ERR))
else:
    # Functions from your module
    extract_data_NMRMNova = nmr.extract_data_NMRMNova
    filter_chemical_shift = nmr.filter_chemical_shift
    mask_regions_with_zeros = nmr.mask_regions_with_zeros
    ref_spectra_to_df = nmr.ref_spectra_to_df
    # Alignment implementations
    align_samples_using_icoshift = getattr(nmr, "align_samples_using_icoshift", None)
    RAFFT_df = getattr(nmr, "RAFFT_df", None)
    PAFFT_df = getattr(nmr, "PAFFT_df", None)
    # Plots and exports (Plotly-based in your file)
    create_nmr_plot = getattr(nmr, "create_nmr_plot", None)
    create_vertical_multiplot = getattr(nmr, "create_vertical_multiplot", None)
    perform_pca_analysis = getattr(nmr, "perform_pca_analysis", None)
    plot_pca_loadings = getattr(nmr, "plot_pca_loadings", None)
    perform_pls_da = getattr(nmr, "perform_pls_da", None)
    plot_plsda_loadings = getattr(nmr, "plot_plsda_loadings", None)
    export_metaboanalyst = getattr(nmr, "export_metaboanalyst", None)
    # Transforms & normalization
    log_transform = getattr(nmr, "log_transform", None)
    sqrt_transform = getattr(nmr, "sqrt_transform", None)
    cbrt_transform = getattr(nmr, "cbrt_transform", None)
    pqn_normalize = getattr(nmr, "pqn_normalize", None)
    z_score_normalize = getattr(nmr, "z_score_normalize", None)
    std_dev_normalize = getattr(nmr, "std_dev_normalize", None)
    median_normalize = getattr(nmr, "median_normalize", None)
    quantile_normalize = getattr(nmr, "quantile_normalize", None)
    min_max_scale = getattr(nmr, "min_max_scale", None)
    standard_scale = getattr(nmr, "standard_scale", None)
    robust_scale = getattr(nmr, "robust_scale", None)
    mean_center = getattr(nmr, "mean_center", None)
    auto_scale = getattr(nmr, "auto_scale", None)
    auto_scale_m = getattr(nmr, "auto_scale_m", None)
    pareto_scale = getattr(nmr, "pareto_scale", None)
    range_scale = getattr(nmr, "range_scale", None)

    if uploads:
        parsed = []
        for f in uploads:
            try:
                df_i = extract_data_NMRMNova(f)  # expects MNova .csv
                stem = os.path.splitext(f.name)[0]
                cols = df_i.columns.tolist()
                if len(cols) == 2:
                    df_i = df_i.rename(columns={cols[1]: stem})
                parsed.append(df_i)
                report_rows.append({"file": f.name, "status": "parsed", "rows": len(df_i)})
            except Exception as e:
                report_rows.append({"file": f.name, "status": f"error: {e}", "rows": 0})

        rep = pd.DataFrame(report_rows)
        if not rep.empty:
            st.subheader("Parsing report")
            show_df(rep)

        if parsed:
            combined = interpolate_common(outer_merge_on_ppm(parsed, "Chemical Shift (ppm)"))

# ------------------ Metadata (optional) --------------------------------
meta_df = None
if meta_file is not None:
    try:
        meta_df = pd.read_csv(meta_file)
    except Exception:
        try:
            meta_file.seek(0)
            meta_df = pd.read_csv(meta_file, sep=";")
        except Exception:
            meta_file.seek(0)
            meta_df = pd.read_csv(meta_file, sep="\t")

# ------------------ Preprocessing controls -----------------------------
st.markdown("---")
st.subheader("Preprocessing")

c1, c2 = st.columns(2)
with c1:
    ppm_min = st.number_input("ppm min (right side)", value=-0.5, step=0.1, format="%.3f")
with c2:
    ppm_max = st.number_input("ppm max (left side)", value=10.0, step=0.1, format="%.3f")

mask_str = st.text_input("Mask ppm regions (start:end; start:end)", value="")  # e.g., 4.70:4.90; 3.30:3.40
regions = parse_mask_regions(mask_str) if mask_str else []

processed0 = None
if combined is not None:
    processed0 = filter_chemical_shift(combined, start_limit=ppm_min, end_limit=ppm_max)
    if regions:
        processed0 = mask_regions_with_zeros(processed0, regions)
    st.caption("Combined/processed (head)")
    show_df(processed0.head(5))

def reference_to_target_ppm(
    df: pd.DataFrame,
    target_ppm: float = 0.0,
    search_window: tuple[float, float] = (-0.05, 0.05),
    axis_col: str = "Chemical Shift (ppm)",
    use_abs: bool = True,
    fill_edges: bool = True
) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        return df, {}
    out = df.copy()
    out = out.sort_values(axis_col).reset_index(drop=True)
    ppm = out[axis_col].astype(float).values

    lo = target_ppm + float(search_window[0])
    hi = target_ppm + float(search_window[1])
    if lo > hi:
        lo, hi = hi, lo
    mask = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 2:
        raise ValueError(f"Search window too narrow or outside data: [{lo}, {hi}] ppm.")

    offsets = {}
    for col in [c for c in out.columns if c != axis_col]:
        y = pd.to_numeric(out[col], errors="coerce").values
        y_win = y[mask]; ppm_win = ppm[mask]
        if np.all(~np.isfinite(y_win)) or y_win.size == 0:
            offsets[col] = 0.0
            continue
        idx = int(np.nanargmax(np.abs(y_win) if use_abs else y_win))
        peak_ppm = float(ppm_win[idx])
        shift = float(target_ppm - peak_ppm)  # +shift moves to higher ppm
        offsets[col] = shift
        src_x = ppm - shift
        new_y = np.interp(ppm, src_x, y, left=np.nan, right=np.nan)
        if fill_edges:
            s = pd.Series(new_y)
            new_y = s.fillna(method="bfill").fillna(method="ffill").values
        out[col] = new_y
    return out, offsets


# ------------------ Referencing (anchor a peak to chosen ppm) ------------------
st.markdown("---")
st.subheader("Referencing (anchor a peak to chosen ppm)")

# restore referenced copy if present
referenced_df = st.session_state.get("referenced_df")

if processed0 is None and referenced_df is None:
    st.caption("Load and preprocess data to enable referencing.")
else:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        target_ppm = st.number_input("Target ppm (e.g., 0.00, 2.50)", value=0.00, step=0.01, format="%.2f")
    with c2:
        halfwin = st.number_input("Search half-window (ppm)", value=0.35, step=0.05, format="%.2f")
    with c3:
        use_abs = st.checkbox("Use |peak| (robust to phase)", value=True)
    with c4:
        do_clear = st.button("Clear referenced")

    do_ref = st.button("Reference to target ppm")
    if do_clear:
        for k in ("referenced_df", "nmr_final_df", "postref_ppm_FULL", "postref_ppm_FULL_SIG", "bounds_postref_ppm"):
            st.session_state.pop(k, None)
        referenced_df = None
        st.info("Cleared referenced data.")

    if do_ref:
        try:
            base = processed0 if processed0 is not None else referenced_df
            if base is None:
                st.warning("Nothing to reference yet.")
            else:
                if hasattr(nmr, "reference_to_target_ppm"):
                    referenced_df, offsets = nmr.reference_to_target_ppm(
                        base.copy(),
                        target_ppm=float(target_ppm),
                        search_window=(-halfwin, halfwin),
                        axis_col="Chemical Shift (ppm)",
                        use_abs=bool(use_abs)
                    )
                else:
                    referenced_df, offsets = reference_to_target_ppm(
                        base.copy(),
                        target_ppm=float(target_ppm),
                        search_window=(-halfwin, halfwin),
                        axis_col="Chemical Shift (ppm)",
                        use_abs=bool(use_abs)
                    )
                st.session_state["referenced_df"] = referenced_df
                st.success(f"Referenced to {float(target_ppm):.2f} ppm.")
                with st.expander("Applied ppm shifts per sample"):
                    show_df(pd.DataFrame.from_dict(offsets, orient="index", columns=["ppm_shift"]))
        except Exception as e:
            st.error(f"Referencing failed: {e}")

    if referenced_df is not None:
        st.caption("Referenced (head)")
        show_df(referenced_df.head(5))

# ---------- helper: generic clipping UI for any ppm-based matrix ----------
def clip_ppm_range_ui(df: pd.DataFrame, key_prefix: str, axis_col: str = "Chemical Shift (ppm)"):
    """
    Interactive clipper that persists full matrix and bounds in session_state.
    Returns the currently clipped DataFrame.
    """
    if df is None or df.empty:
        st.caption("Load data to enable ppm clipping.")
        return df

    # Build a light signature of df to know when upstream changed
    def _sig(d: pd.DataFrame, xcol: str):
        cols = tuple(d.columns)
        n = len(d)
        lo = float(d[xcol].iloc[0]) if n else np.nan
        hi = float(d[xcol].iloc[-1]) if n else np.nan
        return (cols, n, lo, hi)

    full_key  = f"{key_prefix}_FULL"
    sig_key   = f"{key_prefix}_FULL_SIG"
    bounds_key = f"bounds_{key_prefix}"

    # ensure ascending axis internally
    dfx = df.sort_values(axis_col).reset_index(drop=True)
    current_sig = _sig(dfx, axis_col)

    if (full_key not in st.session_state) or (sig_key not in st.session_state) or (st.session_state[sig_key] != current_sig):
        st.session_state[full_key] = dfx.copy()
        st.session_state[sig_key]  = current_sig
        lo_full = float(dfx[axis_col].min())
        hi_full = float(dfx[axis_col].max())
        st.session_state[bounds_key] = (lo_full, hi_full)

    full_df = st.session_state[full_key]
    lo_full = float(full_df[axis_col].min())
    hi_full = float(full_df[axis_col].max())

    saved_lo, saved_hi = st.session_state.get(bounds_key, (lo_full, hi_full))

    # ACTIVE (clipped) df is auto-produced every rerun
    lo_act, hi_act = min(saved_lo, saved_hi), max(saved_lo, saved_hi)
    active_df = nmr.filter_chemical_shift(full_df, start_limit=lo_act, end_limit=hi_act)

    # UI
    col_a, col_b = st.columns(2)
    with col_a:
        start_ppm = st.number_input("Start ppm", value=float(saved_lo), min_value=lo_full, max_value=hi_full, step=0.01, format="%.3f")
    with col_b:
        end_ppm   = st.number_input("End ppm",   value=float(saved_hi), min_value=lo_full, max_value=hi_full, step=0.01, format="%.3f")

    # Clamp & buttons
    start_ppm = float(min(max(start_ppm, lo_full), hi_full))
    end_ppm   = float(min(max(end_ppm,   lo_full), hi_full))

    b1, b2, _ = st.columns([1, 1, 2])
    apply_clip = b1.button("Apply clipping", key=f"apply_{key_prefix}")
    reset_clip = b2.button("Reset to full range", key=f"reset_{key_prefix}")

    if reset_clip:
        st.session_state[bounds_key] = (lo_full, hi_full)
        active_df = nmr.filter_chemical_shift(full_df, start_limit=lo_full, end_limit=hi_full)
        st.success(f"Reset ppm range to [{lo_full:.3f}, {hi_full:.3f}]")

    if apply_clip:
        lo, hi = (start_ppm, end_ppm) if start_ppm <= end_ppm else (end_ppm, start_ppm)
        tmp = nmr.filter_chemical_shift(full_df, start_limit=lo, end_limit=hi)
        if tmp is None or tmp.empty:
            st.warning("Clipping produced an empty table. No changes applied.")
        else:
            st.session_state[bounds_key] = (lo, hi)
            active_df = tmp
            st.success(f"Clipped to ppm ∈ [{lo:.3f}, {hi:.3f}] — {len(tmp)} points.")

    cur_lo, cur_hi = st.session_state[bounds_key]
    st.caption(f"Current ppm window: [{cur_lo:.3f}, {cur_hi:.3f}]")
    with st.expander("Preview (head)"):
        show_df(active_df.head(10))

    return active_df

# ---- Optional SciPy for KDE (faster/robuster if available) ----
try:
    from scipy import stats as sstats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def _as_numeric(df: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == axis_col: 
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _silverman_bw(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 1.0
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(s, iqr/1.349) if (s > 0 and iqr > 0) else (s if s > 0 else 1.0)
    return 0.9 * sigma * (n ** (-1/5))

def _kde1d(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    if HAVE_SCIPY:
        kde = sstats.gaussian_kde(x, bw_method=(bw if bw else "scott"))
        return kde(grid)
    # NumPy-only Gaussian KDE
    h = float(bw) if (bw and bw > 0) else _silverman_bw(x)
    # avoid divide-by-zero
    h = h if h > 0 else 1.0
    # (n,1) broadcasting
    diffs = (grid[None, :] - x[:, None]) / h
    dens = np.exp(-0.5 * diffs**2) / (np.sqrt(2*np.pi) * h)
    return dens.sum(axis=0) / x.size

# ------------------ Clip ppm Range (after Referencing) ------------------
st.markdown("---")
st.subheader("Clip ppm Range (after Referencing)")

src = st.session_state.get("referenced_df")
if src is None:
    st.caption("Reference spectra first to enable post-reference clipping.")
else:
    clipped = clip_ppm_range_ui(src, key_prefix="postref_ppm")
    # promote to canonical dataset for ALL downstream steps
    old_sig = st.session_state.get("nmr_final_sig")
    new_sig = (tuple(clipped.columns), len(clipped),
               float(clipped["Chemical Shift (ppm)"].min()),
               float(clipped["Chemical Shift (ppm)"].max()))
    st.session_state["referenced_df"] = clipped
    st.session_state["nmr_final_df"] = clipped
    st.session_state["nmr_final_sig"] = new_sig
    if old_sig is None or old_sig != new_sig:
        st.session_state.pop("aligned_df", None)
        st.session_state.pop("normalized_df", None)

# ------------------ Alignment (iCOshift / RAFFT / PAFFT) ---------------
st.markdown("---")
st.subheader("Alignment")

df_align_source = coalesce_df(
    st.session_state.get("nmr_final_df"),
    referenced_df,
    processed0,
)
df_aligned = None

def alignment_controls(available: dict) -> Tuple[str, dict]:
    # Build options dynamically
    opts = ["None"]
    if HAVE_PYICOSHIFT and align_samples_using_icoshift:
        opts.append("iCOshift")
    if RAFFT_df is not None:
        opts.append("RAFFT")
    if PAFFT_df is not None:
        opts.append("PAFFT")
    method = st.selectbox("Method", options=opts, index=0)
    params = {}
    if method == "iCOshift":
        params["n_intervals"] = st.slider("iCOshift intervals", min_value=10, max_value=200, value=50, step=5)
    elif method == "RAFFT":
        params["reference_idx"] = st.number_input("Reference sample index (0-based)", min_value=0, value=0, step=1)
    elif method == "PAFFT":
        params["segSize_ppm"] = st.number_input("PAFFT segSize (ppm)", value=0.02, step=0.01, format="%.3f")
        params["reference_idx"] = st.number_input("Reference sample index (0-based)", min_value=0, value=0, step=1)
    return method, params

def align_df(df: pd.DataFrame, method: str, **kw) -> pd.DataFrame:
    if method == "None" or df is None:
        return df
    if method == "iCOshift" and align_samples_using_icoshift:
        return align_samples_using_icoshift(df, n_intervals=int(kw.get("n_intervals", 50)))
    if method == "RAFFT" and RAFFT_df:
        return RAFFT_df(df, reference_idx=int(kw.get("reference_idx", 0)))
    if method == "PAFFT" and PAFFT_df:
        return PAFFT_df(df, segSize_ppm=float(kw.get("segSize_ppm", 0.02)), reference_idx=int(kw.get("reference_idx", 0)))
    st.warning("Selected alignment method is unavailable; returning input.")
    return df

if df_align_source is None:
    st.info("Load/process/reference data to enable alignment.")
else:
    sample_names = list(df_align_source.columns[1:])
    method, params = alignment_controls({"samples": sample_names})
    do_align = st.button("Run alignment")
    if do_align:
        try:
            df_aligned = align_df(df_align_source.copy(), method, **params)
            st.success(f"Aligned using {method}.")
            st.caption("Aligned (head)")
            show_df(df_aligned.head(5))
        except Exception as e:
            st.error(f"Alignment failed: {e}")

# ------------------ Transforms, Normalization & Scaling ----------------------------
st.markdown("---")
st.subheader("Transforms, Normalization & Scaling")

# choose base matrix safely
working_df = coalesce_df(
    df_aligned,
    st.session_state.get("nmr_final_df"),
    referenced_df,
    processed0,
)
normalized_df = None

if working_df is None:
    st.info("Load/process data to enable normalization/scaling.")
else:
    # ------- 1) TRANSFORM -------
    tform = st.radio(
        "Transform",
        options=["none", "log10(+1)", "sqrt", "cbrt"],
        horizontal=True,
    )

    # ------- 2) NORMALIZATION (intensity normalization across samples) -------
    st.markdown("**Normalization**")
    norm_choice = st.selectbox(
        "Choose a normalization method",
        options=[
            "none",
            "Z-score",
            "PQN",
            "Std-dev",
            "Median=1",
            "Quantile",
        ],
        index=0,
        help="Normalization is applied BEFORE scaling.",
    )

    # (optional) parameters for normalization
    # e.g., target median for 'Median=1' could be added here if needed.

    # ------- 3) SCALING (feature scaling along ppm axis) -------
    st.markdown("**Scaling**")
    scale_choice = st.selectbox(
        "Choose a scaling method",
        options=[
            "none",
            "standard (mean=0, sd=1)",
            "robust (median/IQR)",
            "pareto (mean-centered / sqrt(sd))",
            "range (mean-centered / range)",
            "min-max [0..1]",
            "mean-center",
            "auto-scale (classic)",
            "auto-scale (sklearn)",
        ],
        index=0,
        help="Scaling is applied AFTER normalization.",
    )

    # ---- helpers ----
    def apply_transform(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        if tform == "log10(+1)" and log_transform: return log_transform(df_out, constant=1)
        if tform == "sqrt" and sqrt_transform:     return sqrt_transform(df_out)
        if tform == "cbrt" and cbrt_transform:     return cbrt_transform(df_out)
        return df_out

    def apply_normalization(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        if norm_choice == "Z-score" and z_score_normalize:      return z_score_normalize(df_out)
        if norm_choice == "PQN" and pqn_normalize:              return pqn_normalize(df_out)
        if norm_choice == "Std-dev" and std_dev_normalize:      return std_dev_normalize(df_out)
        if norm_choice == "Median=1" and median_normalize:      return median_normalize(df_out)
        if norm_choice == "Quantile" and quantile_normalize:    return quantile_normalize(df_out)
        return df_out

    def apply_scaling(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        if scale_choice == "standard (mean=0, sd=1)" and standard_scale:  return standard_scale(df_out)
        if scale_choice == "robust (median/IQR)" and robust_scale:        return robust_scale(df_out)
        if scale_choice == "pareto (mean-centered / sqrt(sd))" and pareto_scale:  return pareto_scale(df_out)
        if scale_choice == "range (mean-centered / range)" and range_scale:       return range_scale(df_out)
        if scale_choice == "min-max [0..1]" and min_max_scale:            return min_max_scale(df_out)
        if scale_choice == "mean-center" and mean_center:                  return mean_center(df_out)
        if scale_choice == "auto-scale (classic)" and auto_scale:          return auto_scale(df_out)
        if scale_choice == "auto-scale (sklearn)" and auto_scale_m:        return auto_scale_m(df_out)
        return df_out

    # ---- pipeline: Transform -> Normalize -> Scale ----
    stage1 = apply_transform(working_df)
    stage2 = apply_normalization(stage1)
    normalized_df = apply_scaling(stage2)

    st.success("Transforms/normalization/scaling applied (in order).")
    show_df(normalized_df.head(5))

# ------------------ QC — Overlayed density curves (visual normality) ------------------
st.markdown("---")
st.subheader("QC — Overlayed density curves")

axis_col = "Chemical Shift (ppm)"

# Pick which dataset to inspect
overlay_stage = st.radio(
    "Dataset to overlay",
    ["Normalized (before scaling)", "Scaled (after scaling)"],
    index=0, horizontal=True
)
overlay_df = st.session_state.get("nmr_stage2_normalized") if overlay_stage.startswith("Normalized") \
             else st.session_state.get("nmr_stage3_scaled")

if not isinstance(overlay_df, pd.DataFrame) or overlay_df.empty:
    st.info("Nothing to overlay yet — run Transform/Normalization/Scaling first.")
else:
    overlay_df = _as_numeric(overlay_df, axis_col)

    tabs = st.tabs(["By sample (across ppm)", "By feature/ppm (across samples)"])

    # ---------------- By sample ----------------
    with tabs[0]:
        st.caption("Each curve is one sample’s intensity distribution across ppm.")
        z_each = st.checkbox("Z-score each sample before KDE (recommended)", value=True, key="kde_z_sample")
        max_curves = st.slider("Max samples to overlay", 3, 30, min(10, overlay_df.shape[1]-1), 1)
        opacity = st.slider("Curve opacity", 0.1, 1.0, 0.4, 0.05)

        samples_all = [c for c in overlay_df.columns if c != axis_col]
        pick = st.multiselect("Samples", samples_all, default=samples_all[:max_curves])
        if not pick:
            st.warning("Select at least one sample.")
        else:
            # Grid: standard range if z-scored, else based on pooled quantiles
            if z_each:
                grid = np.linspace(-4, 4, 600)
                ref_pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * grid**2)  # N(0,1)
            else:
                X = np.hstack([overlay_df[s].to_numpy(dtype=float) for s in pick])
                lo, hi = np.nanpercentile(X, [0.5, 99.5])
                grid = np.linspace(lo, hi, 600)
                ref_pdf = None

            fig = go.Figure()
            for s in pick:
                x = overlay_df[s].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if x.size < 8:
                    continue
                if z_each:
                    sd = np.std(x, ddof=1)
                    if sd == 0:
                        continue
                    x = (x - np.mean(x)) / sd
                y = _kde1d(x, grid)
                fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", name=str(s), opacity=opacity))

            if z_each:
                fig.add_trace(go.Scatter(x=grid, y=ref_pdf, mode="lines", name="N(0,1)", line=dict(width=3)))
            fig.update_layout(
                title=f"Overlayed density — {overlay_stage} (by sample)",
                xaxis_title=("z-score" if z_each else "Intensity"),
                yaxis_title="Density"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- By feature/ppm ----------------
    with tabs[1]:
        st.caption("Each curve is one feature’s (ppm) distribution across samples.")
        z_each_f = st.checkbox("Z-score each feature before KDE (recommended)", value=True, key="kde_z_feat")

        method = st.selectbox("Feature selection",
                              ["Top variance", "Random", "Manual ppm list"], index=0)
        how_many = st.slider("Number of features to overlay", 3, 50, 10, 1)

        # Build candidate ppm indices
        ppm_vals = overlay_df[axis_col].astype(float).to_numpy()
        M = overlay_df.drop(columns=[axis_col]).to_numpy(dtype=float)  # rows=ppm, cols=samples
        var_by_feat = np.nanvar(M, axis=1)

        if method == "Top variance":
            idxs = np.argsort(var_by_feat)[::-1][:how_many]
        elif method == "Random":
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(ppm_vals), size=min(how_many, len(ppm_vals)), replace=False)
        else:
            ppm_list = st.text_input("ppm list (comma-separated; nearest rows will be used)", value="")
            cand = []
            for tok in ppm_list.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    val = float(tok)
                    cand.append(int(np.argmin(np.abs(ppm_vals - val))))
                except Exception:
                    pass
            idxs = np.array(cand[:how_many], dtype=int)

        if idxs.size == 0:
            st.warning("No features selected.")
        else:
            # Grid
            if z_each_f:
                grid = np.linspace(-4, 4, 600)
                ref_pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * grid**2)
            else:
                # pool across chosen features
                pool = []
                for i in idxs:
                    v = M[i, :]
                    v = v[np.isfinite(v)]
                    if v.size: pool.append(v)
                if pool:
                    pool = np.concatenate(pool)
                    lo, hi = np.nanpercentile(pool, [0.5, 99.5])
                else:
                    lo, hi = -1.0, 1.0
                grid = np.linspace(lo, hi, 600)
                ref_pdf = None

            opacity_f = st.slider("Curve opacity (features)", 0.1, 1.0, 0.5, 0.05)

            figf = go.Figure()
            for i in idxs:
                v = M[i, :]
                v = v[np.isfinite(v)]
                if v.size < 8:
                    continue
                if z_each_f:
                    sd = np.std(v, ddof=1)
                    if sd == 0:
                        continue
                    v = (v - np.mean(v)) / sd
                y = _kde1d(v, grid)
                figf.add_trace(go.Scatter(
                    x=grid, y=y, mode="lines",
                    name=f"{ppm_vals[i]:.4f} ppm", opacity=opacity_f
                ))

            if z_each_f:
                figf.add_trace(go.Scatter(x=grid, y=ref_pdf, mode="lines", name="N(0,1)", line=dict(width=3)))

            figf.update_layout(
                title=f"Overlayed density — {overlay_stage} (by feature/ppm)",
                xaxis_title=("z-score" if z_each_f else "Intensity"),
                yaxis_title="Density"
            )
            st.plotly_chart(figf, use_container_width=True)


# ------------------ QC: Normality of Normalized & Scaled data ------------------
st.markdown("---")
st.subheader("QC — Normality check (histogram + normal curve + Q–Q plot)")

axis_col = "Chemical Shift (ppm)"

# SciPy optional
try:
    from scipy import stats as sstats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Deterministic inverse-normal (Acklam) for when SciPy isn't available
def _norm_ppf(p: np.ndarray) -> np.ndarray:
    if HAVE_SCIPY:
        return sstats.norm.ppf(p)
    p = np.asarray(p, dtype=float)
    # Coefficients for Acklam's approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow  = 0.02425
    phigh = 1.0 - plow
    x = np.empty_like(p)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(p[mask]))
        x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q*q
        x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1.0 - p[mask]))
        x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return x

# pick stage
stage_choice = st.radio(
    "Dataset to inspect",
    ["Normalized (before scaling)", "Scaled (after scaling)"],
    horizontal=True,
    index=0,
)

stage_map = {
    "Normalized (before scaling)": st.session_state.get("nmr_stage2_normalized"),
    "Scaled (after scaling)":      st.session_state.get("nmr_stage3_scaled"),
}
stage_df = stage_map.get(stage_choice)

def _as_numeric(df, axis=axis_col):
    df = df.copy()
    for c in df.columns:
        if c == axis: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if not isinstance(stage_df, pd.DataFrame) or stage_df.empty:
    st.info("Nothing to inspect yet — run Transform/Normalization/Scaling first.")
else:
    stage_df = _as_numeric(stage_df, axis_col)

    mode = st.radio(
        "Distribution of",
        ["intensities across ppm (per sample)", "intensities across samples (per feature/ppm)"],
        horizontal=False,
    )

    # choose vector x to test
    x = None
    label = ""
    if mode.startswith("intensities across ppm"):
        samples = [c for c in stage_df.columns if c != axis_col]
        sample_sel = st.selectbox("Sample", samples, index=0)
        x = stage_df[sample_sel].astype(float).to_numpy()
        label = f"{stage_choice} — Sample: {sample_sel}"
    else:
        ppm_vals = stage_df[axis_col].astype(float).values
        ppm_sel = st.number_input("ppm to inspect (nearest row)", value=float(np.median(ppm_vals)), step=0.01, format="%.3f")
        idx = int(np.argmin(np.abs(ppm_vals - float(ppm_sel))))
        row = stage_df.iloc[idx]
        x = row.drop(labels=[axis_col]).astype(float).to_numpy()
        label = f"{stage_choice} — Feature @ {float(ppm_vals[idx]):.4f} ppm"

    x = x[np.isfinite(x)]
    x = x[~np.isnan(x)]
    n = int(x.size)

    if n < 8:
        st.warning("Not enough points for normality checks (need ≥ 8).")
    else:
        mu  = float(np.mean(x))
        sd  = float(np.std(x, ddof=1)) if n > 1 else 0.0
        skew = float(pd.Series(x).skew())
        kurt = float(pd.Series(x).kurt())

        # Histogram + fitted normal curve
        bins = st.slider("Bins", min_value=20, max_value=200, value=60, step=10)
        xx = np.linspace(np.min(x), np.max(x), 400)
        sigma = sd if sd > 0 else 1.0
        pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xx - mu)/sigma)**2)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=bins, name="Data"))
        fig_hist.add_trace(go.Scatter(x=xx, y=pdf, mode="lines", name="Normal fit (μ, σ)"))
        fig_hist.update_layout(title=f"Histogram + Normal curve — {label}",
                               xaxis_title="Intensity", yaxis_title="Density")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Q–Q plot
        p = (np.arange(1, n+1) - 0.5) / n
        theo = _norm_ppf(p)
        x_sorted = np.sort(x)
        a = np.polyfit(theo, x_sorted, 1)  # least-squares line
        fit_y = a[0]*theo + a[1]

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theo, y=x_sorted, mode="markers", name="Empirical"))
        fig_qq.add_trace(go.Scatter(x=theo, y=fit_y, mode="lines", name="Fit line"))
        fig_qq.update_layout(title=f"Q–Q plot vs Normal — {label}",
                             xaxis_title="Theoretical quantiles (N(0,1))",
                             yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

        # Tests (when SciPy is present)
        if HAVE_SCIPY and sd > 0:
            sh_stat, sh_p = sstats.shapiro(x) if n <= 5000 else (np.nan, np.nan)
            try:
                k2_stat, k2_p = sstats.normaltest(x)  # D'Agostino & Pearson
            except Exception:
                k2_stat, k2_p = (np.nan, np.nan)
            ad = sstats.anderson(x, dist='norm')
            st.markdown(
                f"**Normality tests (n={n}):**  \n"
                f"- Shapiro–Wilk p = {sh_p:.3g} (n≤5000 only)  \n"
                f"- D’Agostino K² p = {k2_p:.3g}  \n"
                f"- Anderson–Darling A² = {ad.statistic:.3g} (5% crit ≈ {ad.critical_values[2]:.3g})  \n"
                f"**Moments:** μ={mu:.4g}, σ={sd:.4g}, skew={skew:.3g}, kurtosis={kurt:.3g}"
            )
        else:
            st.markdown(
                f"**Moments (n={n}):** μ={mu:.4g}, σ={sd:.4g}, skew={skew:.3g}, kurtosis={kurt:.3g}  \n"
                f"_Install SciPy to run Shapiro / D’Agostino / Anderson tests._"
            )



# ---- pipeline: Transform -> Normalize -> Scale ----
stage1 = apply_transform(working_df)
stage2 = apply_normalization(stage1)     # <-- "Normalized" (pre-scaling)
normalized_df = apply_scaling(stage2)    # <-- "Scaled" (post-scaling)

# keep for QC panel
st.session_state["nmr_stage1_transformed"] = stage1
st.session_state["nmr_stage2_normalized"] = stage2
st.session_state["nmr_stage3_scaled"] = normalized_df

st.success("Transforms/normalization/scaling applied (in order).")
show_df(normalized_df.head(5))

# ------------------ QC: Normalization & Scaling checks ------------------
st.markdown("---")
st.subheader("QC — Check Normalized vs Scaled (samples & features)")

axis_col = "Chemical Shift (ppm)"

# ---------- helpers ----------
def _as_numeric(df, axis=axis_col):
    out = df.copy()
    for c in out.columns:
        if c == axis: 
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def sample_stats(df: pd.DataFrame, axis=axis_col) -> pd.DataFrame:
    """Per-sample stats across ppm (columns = samples)."""
    X = _as_numeric(df, axis).drop(columns=[axis], errors="ignore")
    ssum = X.sum(axis=0)
    smean = X.mean(axis=0)
    sstd = X.std(axis=0, ddof=1)
    smed = X.median(axis=0)
    smin = X.min(axis=0)
    smax = X.max(axis=0)
    l2 = np.sqrt((X**2).sum(axis=0))
    nz = (X != 0).sum(axis=0)
    nz_pct = 100.0 * nz / X.shape[0]
    out = pd.DataFrame({
        "Sample": X.columns,
        "Sum": ssum.values,
        "Mean": smean.values,
        "Std": sstd.values,
        "Median": smed.values,
        "Min": smin.values,
        "Max": smax.values,
        "L2_norm": l2.values,
        "%NonZero": nz_pct.values,
    })
    return out.sort_values("Sample").reset_index(drop=True)

def feature_stats(df: pd.DataFrame, axis=axis_col) -> pd.DataFrame:
    """Per-feature (ppm row) stats across samples (rows = ppm)."""
    X = _as_numeric(df, axis)
    ppm = X[axis].astype(float).values
    M = X.drop(columns=[axis], errors="ignore").values  # rows=ppm, cols=samples
    mean_ = np.nanmean(M, axis=1)
    std_  = np.nanstd(M,  axis=1, ddof=1)
    # CV guard for near-zero means
    eps = np.where(np.abs(mean_) < 1e-12, 1.0, mean_)
    cv = std_ / np.abs(eps)
    min_ = np.nanmin(M, axis=1)
    max_ = np.nanmax(M, axis=1)
    nz   = np.sum(M != 0, axis=1)
    nz_pct = 100.0 * nz / M.shape[1] if M.shape[1] else np.zeros_like(nz, dtype=float)
    return pd.DataFrame({
        axis: ppm,
        "Mean": mean_,
        "Std": std_,
        "CV": cv,
        "Min": min_,
        "Max": max_,
        "%NonZero": nz_pct,
    }).sort_values(axis, ascending=True).reset_index(drop=True)

def _pick_stage(which: str) -> pd.DataFrame | None:
    if which == "Normalized (before scaling)":
        return st.session_state.get("nmr_stage2_normalized")
    if which == "Scaled (after scaling)":
        return st.session_state.get("nmr_stage3_scaled")
    if which == "Transformed (pre-normalization)":
        return st.session_state.get("nmr_stage1_transformed")
    return None

# ---------- UI ----------
which_stage = st.radio(
    "Dataset to inspect",
    options=["Normalized (before scaling)", "Scaled (after scaling)", "Transformed (pre-normalization)"],
    index=0,
    horizontal=True
)

qc_df = _pick_stage(which_stage)

if not isinstance(qc_df, pd.DataFrame) or qc_df.empty:
    st.info("Nothing to inspect yet. Run transform/normalization/scaling first.")
else:
    # tables
    st.markdown("**Per-sample statistics**")
    sstats = sample_stats(qc_df, axis=axis_col)
    show_df(sstats)

    st.markdown("**Per-feature (ppm) statistics**")
    fstats = feature_stats(qc_df, axis=axis_col)
    # Let users focus on the most variable features
    top_n = st.slider("Show top-N features by CV", 2, 5, 5, 1)
    top_feat = fstats.reindex(fstats["CV"].abs().sort_values(ascending=False).index).head(top_n)
    show_df(top_feat)

    # quick plots
    t1, t2, t3, t4 = st.tabs(["Sample totals", "Sample correlation", "Feature CV", "Feature variance vs ppm"])
    with t1:
        fig = px.bar(sstats, x="Sample", y="Sum", title="Sample total intensity (area/sum)")
        fig.update_layout(xaxis_title="Sample", yaxis_title="Sum")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        # correlation on samples
        X = _as_numeric(qc_df, axis_col).drop(columns=[axis_col], errors="ignore")
        corr = X.corr(method="pearson")
        fig = px.imshow(corr, title="Sample–Sample correlation (Pearson)", aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        fig = px.histogram(fstats, x="CV", nbins=60, title="Feature CV distribution")
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        fig = px.scatter(fstats, x=axis_col, y="Std", title="Feature standard deviation vs ppm")
        fig.update_xaxes(autorange="reversed")  # NMR look
        st.plotly_chart(fig, use_container_width=True)


# ------------------ STOCSY (ppm axis) ----------------------------------
st.markdown("---")
st.subheader("STOCSY (ppm axis)")

def stocsy_linear(target_ppm: float, df_wide: pd.DataFrame, axis_col="Chemical Shift (ppm)"):
    """
    Minimal STOCSY: per-ppm correlation vs target ppm across samples.
    df_wide: first col = ppm, other cols = samples
    """
    ppm_axis = df_wide[axis_col].astype(float).values
    idx = int(np.argmin(np.abs(ppm_axis - float(target_ppm))))
    Y = df_wide.iloc[:, 1:].astype(float).values  # points x samples
    y0 = Y[idx, :]
    corr = np.zeros(len(ppm_axis), dtype=float)
    covar = np.zeros(len(ppm_axis), dtype=float)
    for i in range(len(ppm_axis)):
        yi = Y[i, :]
        m = np.isfinite(y0) & np.isfinite(yi)
        if m.sum() < 2: 
            corr[i] = 0.0; covar[i] = 0.0; continue
        a = y0[m]; b = yi[m]
        cov = np.cov(a, b)[0, 1] if m.sum() > 1 else 0.0
        denom = a.std(ddof=1) * b.std(ddof=1)
        corr[i] = float(cov / denom) if denom > 0 else 0.0
        covar[i] = float(cov)
    return ppm_axis, corr, covar

df_for_stocsy = normalized_df if normalized_df is not None else working_df
if df_for_stocsy is None:
    st.caption("Normalize (or at least preprocess) to enable STOCSY.")
else:
    c1, c2 = st.columns([1,1])
    with c1:
        target_ppm = st.number_input("Target ppm", value=0.85, step=0.01, format="%.2f")
    with c2:
        run_stocsy = st.button("Run STOCSY")
    if run_stocsy:
        try:
            ppm_axis, corr, covar = stocsy_linear(float(target_ppm), df_for_stocsy)
            res = pd.DataFrame({"Chemical Shift (ppm)": ppm_axis, "Correlation": corr, "Covariance": covar})
            st.markdown("**Top associations (by |Correlation|):**")
            topn = st.slider("Top-N", min_value=5, max_value=200, value=10, step=5)
            top = res.reindex(res["Correlation"].abs().sort_values(ascending=False).index).head(topn)
            show_df(top)

            figc = px.scatter(res, x="Chemical Shift (ppm)", y="Covariance",
                              color="Correlation", color_continuous_scale="Jet",
                              title=f"STOCSY from {float(target_ppm):.2f} ppm — Covariance colored by Correlation",
                              render_mode="webgl")
            figc.update_traces(marker=dict(size=5))
            figc.update_xaxes(autorange="reversed")
            figc.add_trace(go.Scatter(x=res["Chemical Shift (ppm)"], y=res["Covariance"], mode="lines", name="Covariance"))
            figc.update_layout(xaxis_title="ppm", yaxis_title="Covariance")
            st.plotly_chart(figc, use_container_width=True)

            #figr = px.line(res, x="Chemical Shift (ppm)", y="Correlation", title="Correlation vs ppm")
            #figr.update_xaxes(autorange="reversed")
            #st.plotly_chart(figr, use_container_width=True)

            st.download_button(
                "⬇️ Download STOCSY table (CSV)",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name=f"stocsy_{float(target_ppm):.2f}ppm.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"STOCSY failed: {e}")

# ------------------ Visualizations -------------------------------------
st.markdown("---")
st.subheader("Visualizations")

#viz_df = normalized_df if normalized_df is not None else working_df
viz_df = (
    normalized_df
    if normalized_df is not None
    else (
        df_aligned
        if df_aligned is not None
        else (st.session_state.get("nmr_final_df") or referenced_df or processed0)
    )
)

if viz_df is not None:
    long_df = viz_df.melt(id_vars="Chemical Shift (ppm)", var_name="Sample", value_name="Intensity").dropna(subset=["Intensity"])
    t1, t2, t3 = st.tabs(["Overlay", "Stacked", "Heatmap"])
    with t1:
        fig = px.line(long_df, x="Chemical Shift (ppm)", y="Intensity", color="Sample", title="Overlay spectra")
        # Reverse x to show high→low ppm like NMR
        fig.update_layout(xaxis_title="ppm (downfield → upfield)", yaxis_title="Intensity", legend_title="Sample")
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        samples_sorted = sorted([c for c in viz_df.columns if c != "Chemical Shift (ppm)"])
        step = st.number_input("Stack offset", value=2.0, step=0.5)
        offset_map = {s: i * step for i, s in enumerate(samples_sorted)}
        long_df["Intensity_offset"] = long_df.apply(lambda r: r["Intensity"] + offset_map.get(r["Sample"],0), axis=1)
        fig2 = px.line(long_df, x="Chemical Shift (ppm)", y="Intensity_offset", color="Sample", title="Stacked spectra")
        fig2.update_layout(xaxis_title="ppm", yaxis_title=f"Intensity + offset (step={step})")
        fig2.update_xaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        mat = viz_df[[c for c in viz_df.columns if c != "Chemical Shift (ppm)"]].T.values
        fig3 = go.Figure(data=go.Heatmap(
            z=mat,
            x=viz_df["Chemical Shift (ppm)"].values,
            y=[c for c in viz_df.columns if c != "Chemical Shift (ppm)"],
            coloraxis="coloraxis"
        ))
        fig3.update_layout(title="Intensity heatmap", xaxis_title="ppm", yaxis_title="Sample",
                           coloraxis_colorscale="Viridis")
        fig3.update_xaxes(autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Process some data to enable visualizations.")

# ------------------ Modeling (PCA / PLS-DA or Regression) --------------
st.markdown("---")
st.subheader("Modeling (PCA / PLS)")

model_df = viz_df
if model_df is None:
    st.info("No matrix available for modeling yet.")
elif not SKLEARN_AVAILABLE:
    st.warning(
        "scikit-learn is not installed. Install it to enable PCA/PLS sections.\n"
        "Conda: conda install -c conda-forge scikit-learn\n"
        "Pip:   pip install scikit-learn"
    )
else:
    samples = [c for c in model_df.columns if c != "Chemical Shift (ppm)"]
    X, feats = matrix_to_XY(model_df, samples, axis_col="Chemical Shift (ppm)")

    # Try to auto-pick metadata key that matches sample names
    def _norm_name(s): return str(s).replace(".txt","").strip()
    def _suggest_key_col(meta_df: pd.DataFrame, sample_cols: List[str]) -> str | None:
        norm_samples = {_norm_name(s) for s in sample_cols}
        best_col, best_hits = None, -1
        for col in meta_df.columns:
            vals = {_norm_name(v) for v in meta_df[col].astype(str).tolist()}
            hits = len(norm_samples & vals)
            if hits > best_hits:
                best_col, best_hits = col, hits
        return best_col

    labels, sample_key_col = None, None
    if meta_df is not None and not meta_df.empty:
        guess = _suggest_key_col(meta_df, samples)
        sample_key_col = st.selectbox("Metadata column that matches sample names", options=meta_df.columns.tolist(),
                                      index=(meta_df.columns.tolist().index(guess) if guess in meta_df.columns else 0))
        y_col = st.selectbox("Response/label column", options=["<none>"] + meta_df.columns.tolist(), index=0)
        if y_col != "<none>":
            map_key = meta_df[sample_key_col].astype(str).map(_norm_name)
            sample_to_label = dict(zip(map_key, meta_df[y_col]))
            labels = [sample_to_label.get(_norm_name(s), None) for s in samples]

    # ---------------- PCA ----------------
    with st.expander("PCA", expanded=True):
        n_comp = st.slider("Components", 2, 10, 2, 1)
        scale_pca = st.checkbox("Standardize features before PCA", value=False)

        Xp = X.copy()
        if scale_pca:
            Xp = StandardScaler(with_mean=True, with_std=True).fit_transform(Xp)

        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(Xp)
        expvar = pca.explained_variance_ratio_ * 100.0

        df_scores = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Sample": samples})
        if labels is not None:
            df_scores["Label"] = pd.Series(labels, dtype="object").fillna("NA").astype(str).values
            figp = px.scatter(df_scores, x="PC1", y="PC2", color="Label", hover_name="Sample", title="PCA Scores")
        else:
            figp = px.scatter(df_scores, x="PC1", y="PC2", hover_name="Sample", title="PCA Scores")
        figp.update_layout(xaxis_title=f"PC1 ({expvar[0]:.1f}%)", yaxis_title=f"PC2 ({expvar[1]:.1f}%)")
        st.plotly_chart(figp, use_container_width=True)

        # Loadings on ppm axis — choose any PC
        st.markdown("**PCA X-Loadings**")
        pc_idx = st.number_input("Component to show", min_value=1, max_value=int(n_comp), value=1, step=1)
        load_vec = pca.components_[pc_idx - 1, :]
        show_abs = st.checkbox("Show absolute loadings", value=False)
        smooth_pts = st.number_input("Smooth (moving average, pts)", min_value=1, value=1, step=1)

        y_load = np.abs(load_vec) if show_abs else load_vec
        if smooth_pts > 1:
            y_load = pd.Series(y_load).rolling(window=int(smooth_pts), min_periods=1, center=True).mean().values

        df_load = pd.DataFrame({"Chemical Shift (ppm)": model_df["Chemical Shift (ppm)"].values,
                                "Loading": y_load})
        fig_load = px.line(df_load, x="Chemical Shift (ppm)", y="Loading",
                           title=f"PCA Loading — PC{pc_idx} ({'abs ' if show_abs else ''}loadings)")
        fig_load.update_layout(xaxis_title="ppm", yaxis_title="Loading weight")
        fig_load.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_load, use_container_width=True)

    # ---------------- PLS (PLS-DA or Regression) ----------------
    with st.expander("PLS (PLS-DA / Regression)", expanded=False):
        if labels is None:
            st.info("Select a label/response column in Metadata to enable PLS.")
        else:
            y_raw = np.array(labels, dtype=object)
            y_num_try = pd.to_numeric(y_raw, errors="coerce")
            is_numeric = np.isfinite(y_num_try).all()
            max_lv = int(max(1, min(15, X.shape[1], X.shape[0] - 1)))
            n_comp_pls = st.slider("PLS components", min_value=2, max_value=max_lv, value=min(5, max_lv), step=1)
            scale_X = st.checkbox("Standardize X", value=True)

            X_pls = X.copy()
            if scale_X:
                X_pls = StandardScaler(with_mean=True, with_std=True).fit_transform(X_pls)

            # Helpers to compute R2X / R2Y / Q2
            def _r2x_from_scores_loadings(X_, T_, P_):
                X_hat = T_ @ P_.T
                sse = np.sum((X_ - X_hat) ** 2)
                sst = np.sum((X_ - np.mean(X_, axis=0)) ** 2)
                return float(1.0 - (sse / sst)) if sst > 0 else np.nan

            def _perf_regression(X_, y_, max_a):
                r2x_list, r2y_list, q2_list = [], [], []
                n_splits = max(2, min(5, len(y_)))
                for a in range(1, max_a + 1):
                    pls = PLSRegression(n_components=a).fit(X_, y_)
                    T, P = pls.x_scores_[:, :a], pls.x_loadings_[:, :a]
                    y_hat = pls.predict(X_)
                    r2y = r2_score(y_, y_hat)
                    r2x = _r2x_from_scores_loadings(X_, T, P)
                    # CV
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                    press = sstot = 0.0
                    for tr, te in cv.split(X_):
                        m = PLSRegression(n_components=a).fit(X_[tr], y_[tr])
                        y_pred_te = m.predict(X_[te])
                        press += np.sum((y_[te] - y_pred_te) ** 2)
                        y_mean_tr = np.mean(y_[tr], axis=0, keepdims=True)
                        sstot += np.sum((y_[te] - y_mean_tr) ** 2)
                    q2 = float(1.0 - (press / sstot)) if sstot > 0 else np.nan
                    r2x_list += [r2x]; r2y_list += [r2y]; q2_list += [q2]
                return r2x_list, r2y_list, q2_list

            def _perf_plsda(X_, Y_, y_enc_, max_a):
                r2x_list, r2y_list, q2_list = [], [], []
                counts = np.bincount(y_enc_)
                strat = (len(counts) > 1 and counts.min() >= 2)
                n_splits = max(2, min(5, int(counts.min() if strat else len(y_enc_))))
                for a in range(1, max_a + 1):
                    pls = PLSRegression(n_components=a).fit(X_, Y_)
                    T, P = pls.x_scores_[:, :a], pls.x_loadings_[:, :a]
                    Y_hat = pls.predict(X_)
                    if Y_.shape[1] == 1:
                        r2y = r2_score(Y_, Y_hat)
                    else:
                        r2y = float(np.mean([r2_score(Y_[:, j], Y_hat[:, j]) for j in range(Y_.shape[1])]))
                    r2x = _r2x_from_scores_loadings(X_, T, P)

                    # CV
                    if strat:
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                        splits = cv.split(X_, y_enc_)
                    else:
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                        splits = cv.split(X_)

                    press = sstot = 0.0
                    for tr, te in splits:
                        m = PLSRegression(n_components=a).fit(X_[tr], Y_[tr])
                        Y_pred_te = m.predict(X_[te])
                        press += np.sum((Y_[te] - Y_pred_te) ** 2)
                        Y_mean_tr = np.mean(Y_[tr], axis=0, keepdims=True)
                        sstot += np.sum((Y_[te] - Y_mean_tr) ** 2)
                    q2 = float(1.0 - (press / sstot)) if sstot > 0 else np.nan
                    r2x_list += [r2x]; r2y_list += [r2y]; q2_list += [q2]
                return r2x_list, r2y_list, q2_list

            if is_numeric:
                y = np.asarray(y_num_try, dtype=float).reshape(-1, 1)
                r2x_list, r2y_list, q2_list = _perf_regression(X_pls, y, n_comp_pls)
                pls = PLSRegression(n_components=n_comp_pls).fit(X_pls, y)
                T, P = pls.x_scores_, pls.x_loadings_
                df_scores = pd.DataFrame({"t1": T[:, 0], "t2": (T[:, 1] if T.shape[1] > 1 else np.zeros(T.shape[0])),
                                          "Sample": [str(s) for s in samples], "Label": y_raw})
                fig_scores = px.scatter(df_scores, x="t1", y="t2", color="Label", hover_name="Sample",
                                        title="PLS Regression Scores")
                fig_scores.update_layout(xaxis_title="t[1]", yaxis_title="t[2]")
                st.plotly_chart(fig_scores, use_container_width=True)

                df_load = pd.DataFrame({
                    "Chemical Shift (ppm)": model_df["Chemical Shift (ppm)"].values,
                    "p1": P[:, 0],
                    "p2": (P[:, 1] if P.shape[1] > 1 else np.zeros(P.shape[0]))
                })
                fig_load = go.Figure()
                fig_load.add_trace(go.Scatter(x=df_load["Chemical Shift (ppm)"], y=df_load["p1"], mode="lines", name="p[1]"))
                if P.shape[1] > 1:
                    fig_load.add_trace(go.Scatter(x=df_load["Chemical Shift (ppm)"], y=df_load["p2"], mode="lines", name="p[2]"))
                fig_load.update_layout(title="X-Loadings (p[1], p[2])", xaxis_title="ppm", yaxis_title="Loading")
                fig_load.update_xaxes(autorange="reversed")
                st.plotly_chart(fig_load, use_container_width=True)

                df_perf = pd.DataFrame({"LV": np.arange(1, len(r2x_list)+1), "R²X": r2x_list, "R²Y": r2y_list, "Q²": q2_list})
                df_long = df_perf.melt(id_vars="LV", value_vars=["R²X","R²Y","Q²"], var_name="Metric", value_name="Value")
                fig_perf = px.bar(df_long, x="LV", y="Value", color="Metric", barmode="group", range_y=[0,1],
                                  title="PLS Regression Performance per LV", text="Value")
                fig_perf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_perf.update_layout(yaxis_title="Value (0–1)", xaxis_title="LV")
                st.plotly_chart(fig_perf, use_container_width=True)

            else:
                le = LabelEncoder()
                y_enc = le.fit_transform(y_raw.astype(str))
                classes = le.classes_
                if len(classes) == 2:
                    Y = y_enc.reshape(-1, 1).astype(float)
                else:
                    Y = np.zeros((len(y_enc), len(classes)), dtype=float)
                    Y[np.arange(len(y_enc)), y_enc] = 1.0

                r2x_list, r2y_list, q2_list = _perf_plsda(X_pls, Y, y_enc, n_comp_pls)
                pls = PLSRegression(n_components=n_comp_pls).fit(X_pls, Y)
                T, P = pls.x_scores_, pls.x_loadings_

                df_scores = pd.DataFrame({"t1": T[:, 0], "t2": (T[:, 1] if T.shape[1] > 1 else np.zeros(T.shape[0])),
                                          "Sample": [str(s) for s in samples], "Class": [classes[i] for i in y_enc]})
                fig_scores = px.scatter(df_scores, x="t1", y="t2", color="Class", hover_name="Sample",
                                        title="PLS-DA Scores")
                fig_scores.update_layout(xaxis_title="t[1]", yaxis_title="t[2]")
                st.plotly_chart(fig_scores, use_container_width=True)

                df_load = pd.DataFrame({
                    "Chemical Shift (ppm)": model_df["Chemical Shift (ppm)"].values,
                    "p1": P[:, 0],
                    "p2": (P[:, 1] if P.shape[1] > 1 else np.zeros(P.shape[0]))
                })
                fig_load = go.Figure()
                fig_load.add_trace(go.Scatter(x=df_load["Chemical Shift (ppm)"], y=df_load["p1"], mode="lines", name="p[1]"))
                if P.shape[1] > 1:
                    fig_load.add_trace(go.Scatter(x=df_load["Chemical Shift (ppm)"], y=df_load["p2"], mode="lines", name="p[2]"))
                fig_load.update_layout(title="PLS-DA X-Loadings (p[1], p[2])", xaxis_title="ppm", yaxis_title="Loading")
                fig_load.update_xaxes(autorange="reversed")
                st.plotly_chart(fig_load, use_container_width=True)

                df_perf = pd.DataFrame({"LV": np.arange(1, len(r2x_list)+1), "R²X": r2x_list, "R²Y": r2y_list, "Q²": q2_list})
                df_long = df_perf.melt(id_vars="LV", value_vars=["R²X","R²Y","Q²"], var_name="Metric", value_name="Value")
                fig_perf = px.bar(df_long, x="LV", y="Value", color="Metric", barmode="group", range_y=[0,1],
                                  title="PLS-DA Performance per LV", text="Value")
                fig_perf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_perf.update_layout(yaxis_title="Value (0–1)", xaxis_title="LV")
                st.plotly_chart(fig_perf, use_container_width=True)

# ------------------ Export (ZIP) ---------------------------------------
st.markdown("---")
st.subheader("Export")

final_df = normalized_df if normalized_df is not None else (working_df if working_df is not None else None)
if final_df is None:
    st.info("Nothing to export yet.")
else:
    want_wide = st.checkbox("Include wide matrix (ppm rows, samples columns)", value=True)
    want_long = st.checkbox("Include long format (ppm, Sample, Intensity)", value=False)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if want_wide:
            zf.writestr("processed/nmr_matrix_wide.csv", final_df.to_csv(index=False))
        if want_long:
            long_df = final_df.melt(id_vars="Chemical Shift (ppm)", var_name="Sample", value_name="Intensity")
            zf.writestr("processed/nmr_matrix_long.csv", long_df.to_csv(index=False))
        if meta_df is not None:
            zf.writestr("processed/metadata.csv", meta_df.to_csv(index=False))

    st.download_button(
        "⬇️ Download processed data (.zip)",
        data=zbuf.getvalue(),
        file_name="pyMETAflow_NMR_processed.zip",
        mime="application/zip"
    )

# ------------------ Export (MetaboAnalyst) ---------------------------------------
st.markdown("---")
st.subheader("Export to MetaboAnalyst")

import time  # for default filename timestamps

# 1) pick which matrix to export (prefer fully processed)
export_source = coalesce_df(
    normalized_df,                    # scaled (post-scaling)        ← recommended
    df_aligned,                       # aligned
    st.session_state.get("nmr_final_df"),  # post-reference, post-clip
    referenced_df,
    processed0,
    combined
)

if not isinstance(export_source, pd.DataFrame) or export_source.empty:
    st.info("No matrix available to export yet.")
elif meta_df is None or meta_df.empty:
    st.warning("Load metadata (.csv) to include class labels for MetaboAnalyst.")
else:
    # 2) suggest the metadata column that matches the current sample column names
    axis_col = "Chemical Shift (ppm)"
    sample_cols = [c for c in export_source.columns if c != axis_col]

    def _san(s: str) -> str:
        # must mirror sanitize_string() used in data_processing_NMR.py
        return re.sub(r'[^a-zA-Z0-9_]', '', str(s))

    samples_sanit = {_san(c) for c in sample_cols}

    # Guess sample-id column by maximum sanitized overlap
    guess_id, best_hits = None, -1
    for col in meta_df.columns:
        vals = {_san(v) for v in meta_df[col].astype(str)}
        hits = len(samples_sanit & vals)
        if hits > best_hits:
            best_hits, guess_id = hits, col

    # Guess class column by name (class/group/condition/phenotype…), with your default as backup
    guess_class = None
    for key in ("class", "group", "condition", "phenotype"):
        cands = [c for c in meta_df.columns if key in c.lower()]
        if cands:
            guess_class = cands[0]
            break
    if guess_class is None and "ATTRIBUTE_classification" in meta_df.columns:
        guess_class = "ATTRIBUTE_classification"
    if guess_class is None:
        guess_class = meta_df.columns[-1]

    c1, c2 = st.columns(2)
    with c1:
        sample_id_col = st.selectbox(
            "Metadata column that matches sample columns",
            options=meta_df.columns.tolist(),
            index=(meta_df.columns.tolist().index(guess_id) if guess_id in meta_df.columns else 0),
            help="This column’s values should match your sample column names (after sanitization)."
        )
    with c2:
        class_col = st.selectbox(
            "Metadata class/phenotype column",
            options=meta_df.columns.tolist(),
            index=(meta_df.columns.tolist().index(guess_class) if guess_class in meta_df.columns else 0)
        )

    # Overlap preview
    meta_ids_sanit = meta_df[sample_id_col].astype(str).map(_san)
    overlap = len(samples_sanit & set(meta_ids_sanit))
    st.caption(f"Matched **{overlap}/{len(sample_cols)}** samples after sanitization.")

    # Optional preview of missing samples
    missing = sorted(list(samples_sanit - set(meta_ids_sanit)))
    if missing:
        with st.expander("Samples missing metadata (will be excluded)"):
            show_df(pd.DataFrame({"missing_sample_id_after_sanitize": missing}))

    # Output filename
    default_name = f"metaboanalyst_input_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_name = st.text_input("Output filename", value=default_name)

    # 3) Build & download
    build_btn = st.button("Build MetaboAnalyst CSV")
    if build_btn:
        try:
            if hasattr(nmr, "export_metaboanalyst"):
                # Write to a local file (function returns the feature×samples table used)
                out_path = Path(out_name).name  # keep it simple (no dirs)
                _ = nmr.export_metaboanalyst(
                    export_source,
                    meta_df,
                    sample_id_col=sample_id_col,
                    class_col=class_col,
                    output_file=out_path
                )
                with open(out_path, "rb") as fh:
                    csv_bytes = fh.read()
            else:
                # Fallback: in-memory reimplementation (returns CSV bytes)
                import csv
                orig_cols = list(export_source.columns)
                san_cols  = [orig_cols[0]] + [_san(s) for s in orig_cols[1:]]
                df_al = export_source.copy()
                df_al.columns = san_cols
                sample_cols_san = san_cols[1:]

                meta2 = meta_df.copy()
                meta2[sample_id_col] = meta2[sample_id_col].map(_san)
                meta2[class_col]     = meta2[class_col].map(_san)
                meta_idx = meta2.set_index(sample_id_col)
                cls_series = meta_idx.reindex(sample_cols_san)[class_col]
                valid = cls_series.dropna().index.tolist()

                new_df = df_al[[axis_col] + valid].copy()
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(new_df.columns.tolist())
                w.writerow([""] + cls_series.loc[valid].tolist())
                for i in range(len(new_df)):
                    w.writerow(new_df.iloc[i].values)
                csv_bytes = buf.getvalue().encode("utf-8")

            st.success(f"MetaboAnalyst CSV built ({overlap} samples with class labels).")
            st.download_button(
                "⬇️ Download MetaboAnalyst CSV",
                data=csv_bytes,
                file_name=out_name,
                mime="text/csv"
            )

            # Tiny preview (first few lines; MetaboAnalyst expects first row = header, second row = class)
            try:
                prev = pd.read_csv(io.BytesIO(csv_bytes), header=None, nrows=6)
                st.caption("Preview (first rows):")
                st.dataframe(prev, use_container_width=True)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Export failed: {e}")





st.caption("© pyMETAflow for NMR — Ricardo M. Borges / LAABio-IPPN-UFRJ")
