# data_processing_NMR.py
# by Ricardo Moreira Borges (ricardo_mborges@ufrj.br; IPPN-Universidade Federal do Rio de Janeiro)
#    and Stefan Hermann Kuhn (stefan.kuhn@ut.ee; Universitas Tartuensis)
#
# =============================================================================
# Description:
#    This file provides functions for reading and processing NMR/metabolomics data, including:
#      - Data extraction from MNova exports (.txt -> .csv conversion)
#      - Combining CSV files
#      - Filtering chemical shift/RT ranges and masking unwanted regions
#      - Plotting NMR spectra (overlapping, multiplot, histograms with distributions)
#      - Spectral referencing and alignment (RAFFT, PAFFT, iCOshift, alignment by regions)
#      - Data centering, normalization & scaling methods
#      - Multivariate analysis (PCA, PLS-DA) and variable importance (VIP) calculation
#      - Exporting data in MetaboAnalyst format
#      - STOCSY analysis (interactive and static)
#
#    NO FUNCTIONS HAVE BEEN CHANGED from the original code.
#    Only organization & import grouping have been done.
# =============================================================================

# --------------------------------------------------------------------------
#                           IMPORT STATEMENTS
# --------------------------------------------------------------------------
# Standard Library Imports
import os
import glob
import re
import csv
import math

# Data Handling Imports
import pandas as pd
import numpy as np

# Signal Processing & Statistics Imports
from scipy.signal import correlate, find_peaks
from scipy.stats import norm, stats

# Spectral Alignment Imports
# ---- Optional iCOshift import with graceful fallback ----
HAVE_PYICOSHIFT = True
ICOSHF_CALLABLE = None

try:
    # Try functional API first
    from pyicoshift import icoshift as _icoshift_fn  # type: ignore
    ICOSHF_CALLABLE = _icoshift_fn
except Exception:
    try:
        # Try class API
        from pyicoshift import Icoshift as _IcoshiftCls  # type: ignore

        def _icoshift_from_cls(*args, **kwargs):
            obj = _IcoshiftCls(*args, **kwargs)
            return obj.run() if hasattr(obj, "run") else obj
        ICOSHF_CALLABLE = _icoshift_from_cls
    except Exception:
        HAVE_PYICOSHIFT = False
        ICOSHF_CALLABLE = None


# Visualization Imports
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp

# Data Analysis
from sklearn.decomposition import PCA
import pylab as pl
import math
import mpld3

# --------------------------------------------------------------------------
#               1) DATA EXTRACTION & CSV COMBINATION FUNCTIONS
# --------------------------------------------------------------------------
def extract_data_NMRMNova(file_path):
    """
    Function to import and organize NMR data from an MNova export.
    """
    data = pd.read_csv(file_path, delimiter="\t", on_bad_lines='skip')
    data.rename(columns={data.columns[0]: "Chemical Shift (ppm)"}, inplace=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.dropna(how='all', axis=1, inplace=True)
    data.columns = [col.split("\\")[0].strip() for col in data.columns]
    return data

def combine_and_trim_data_NMRMNova(input_folder, output_folder, retention_time_start, retention_time_end):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            data = extract_data_NMRMNova(file_path)
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_table.csv")
            with open(output_file_path, 'w') as output_file:
                for row in data:
                    output_file.write('\t'.join(row) + '\n')

    file_list = glob.glob(os.path.join(output_folder, '*_table.csv'))
    combined_df = pd.DataFrame()
    for file in file_list:
        column_name = os.path.basename(file).split('_table.csv')[0]
        df = pd.read_csv(file, delimiter='\t', header=None)
        combined_df[column_name] = df.iloc[:, 1]
    axis = df.iloc[:, 0]
    combined_df2 = pd.concat([axis, combined_df], axis=1)
    combined_df2.rename(columns={0: "RT(min)"}, inplace=True)
    start_index = (combined_df2["RT(min)"] - retention_time_start).abs().idxmin()
    end_index = (combined_df2["RT(min)"] - retention_time_end).abs().idxmin()
    combined_df2 = combined_df2.loc[start_index:end_index].copy()
    if not os.path.exists('data'):
        os.mkdir('data')
    combined_df2.to_csv(os.path.join(output_folder, "combined_data.csv"), sep=";", index=False)
    return combined_df2

# --------------------------------------------------------------------------
#               2) FILTERING & MASKING FUNCTIONS
# --------------------------------------------------------------------------
def filter_chemical_shift(data, start_limit=0.5, end_limit=10):
    """
    Filters the DataFrame to keep only rows where the values in the axis column are between start_limit and end_limit.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        start_limit (float): The lower bound of the Chemical Shift range.
        end_limit (float): The upper bound of the Chemical Shift range.
        axis_column (str, optional): The column name representing the Chemical Shift axis.
                                     If None, the function uses the first column.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    filtered_data = data[(data["Chemical Shift (ppm)"] >= start_limit) & (data["Chemical Shift (ppm)"] <= end_limit)]
    return filtered_data

def mask_regions_with_zeros(data, regions):
    """
    Sets values to zero for all rows where the value in the axis column is between region_start and region_end.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        region_start (float): The starting value of the region.
        region_end (float): The ending value of the region.
        axis_column (str, optional): The column name representing the axis (e.g., retention time).
                                     If None, the function uses the first column of the DataFrame.
    
    Returns:
        pd.DataFrame: A modified DataFrame with values in the specified region set to zero (for all columns except the axis).
    """
    modified_data = data.copy()
    for start, end in regions:
        mask = (modified_data["Chemical Shift (ppm)"] >= start) & (modified_data["Chemical Shift (ppm)"] <= end)
        modified_data.loc[mask, modified_data.columns[1:]] = 0
    return modified_data

# --------------------------------------------------------------------------
#               3) PLOTTING FUNCTIONS FOR NMR SPECTRA
# --------------------------------------------------------------------------
def create_nmr_plot(dataframe, 
                    x_axis_col='Chemical Shift (ppm)', 
                    start_column=1, 
                    end_column=None, 
                    title='NMR Spectra Overlapping',
                    xaxis_title='Chemical Shift (ppm)',
                    yaxis_title='Intensity',
                    legend_title='Samples',
                    output_dir='images', 
                    output_file='nmr_spectra_overlapping.html',
                    show_fig=False):
    if end_column is None:
        end_column = len(dataframe.columns) - 1
    fig = go.Figure()
    for column in dataframe.columns[start_column:end_column + 1]:
        fig.add_trace(go.Scatter(
            x=dataframe[x_axis_col],
            y=dataframe[column],
            mode='lines',
            name=column
        ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode='closest',
        xaxis=dict(autorange='reversed')
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.write_html(output_path)
    print(f"Plot saved as: {output_path}")
    if show_fig:
        fig.show()

def create_vertical_multiplot(dataframes, titles,
                              x_axis_col='Chemical Shift (ppm)', 
                              start_column=1, end_column=None, 
                              xaxis_title='Chemical Shift (ppm)', 
                              yaxis_title='Intensity', 
                              legend_title='Samples',
                              output_dir='images', 
                              output_file='aligned_nmr_multiplot.html',
                              show_fig=True):
    n = len(dataframes)
    fig = sp.make_subplots(rows=n, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=titles)
    for i, df in enumerate(dataframes, start=1):
        x = df[x_axis_col].values
        if end_column is None:
            intensity_cols = df.columns[start_column:]
        else:
            intensity_cols = df.columns[start_column:end_column+1]
        for col in intensity_cols:
            fig.add_trace(
                go.Scattergl(x=x, y=df[col].values, mode='lines', name=col),
                row=i, col=1
            )
        fig.update_yaxes(title_text=yaxis_title, row=i, col=1)
    fig.update_layout(
        title="Comparison of Aligned NMR Spectra",
        xaxis_title=xaxis_title,
        legend_title=legend_title,
        height=500 * n,
        margin=dict(t=100),
        xaxis=dict(autorange='reversed')
    )
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, output_file))
    if show_fig:
        fig.show()
    return fig

def plot_histogram_with_distribution(data, output_dir='images', file_name='histogram_with_distribution_curve.html', log_scale=False, x_range=None):
    melted_df = data.melt(value_name="Normalized Value")
    os.makedirs(output_dir, exist_ok=True)
    mean_val = melted_df["Normalized Value"].mean()
    std_val = melted_df["Normalized Value"].std()
    x_values = np.linspace(melted_df["Normalized Value"].min(), melted_df["Normalized Value"].max(), 100)
    normal_curve = norm.pdf(x_values, mean_val, std_val)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=melted_df["Normalized Value"], 
        histnorm='probability density', 
        opacity=0.6,
        name='Normalized Data'
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=normal_curve,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Normal Distribution Curve'
    ))
    fig.update_layout(
        title='Histogram with Normal Distribution Curve',
        xaxis_title='Normalized Value',
        yaxis_title='Density',
        showlegend=True
    )
    if log_scale:
        fig.update_xaxes(type="log")
    if x_range:
        fig.update_xaxes(range=x_range)
    fig.write_html(os.path.join(output_dir, file_name))
    return fig

# --------------------------------------------------------------------------
#               4) SPECTRAL REFERENCING & ALIGNMENT FUNCTIONS
# --------------------------------------------------------------------------
def ref_spectra_to_df(df, thresh=0.01, offsetppm=None, interactive=True, testThreshold=False, xlim=(-0.7, 0.7)):
    axis_col = "Chemical Shift (ppm)"
    if df.columns[0] != axis_col:
        raise ValueError(f"The first column must be named '{axis_col}'")
    ppm_orig = df[axis_col].values
    sample_cols = list(df.columns[1:])
    offsets = {}
    sample1 = sample_cols[0]
    intensity1 = df[sample1].values
    thresh_val = thresh * np.max(intensity1)
    peaks, _ = find_peaks(intensity1, height=thresh_val)
    candidate_mask = (ppm_orig[peaks] >= xlim[0]) & (ppm_orig[peaks] <= xlim[1])
    candidate_peaks = peaks[candidate_mask]
    if len(candidate_peaks) == 0:
        print(f"No candidate peaks found in {sample1} within {xlim}. No referencing applied for this spectrum.")
        offset1 = 0.0
    else:
        if testThreshold:
            plt.figure()
            plt.plot(ppm_orig, intensity1, 'k-', label=sample1)
            plt.plot(ppm_orig[candidate_peaks], intensity1[candidate_peaks], 'ro', label='Candidates')
            plt.xlim(xlim)
            plt.gca().invert_xaxis()
            plt.title(f"Candidate Peaks for {sample1} (Threshold Test)")
            plt.xlabel("Chemical Shift (ppm)")
            plt.ylabel("Intensity")
            plt.legend()
            plt.show()
            return None, None
        if offsetppm is not None:
            diff = np.abs(ppm_orig[candidate_peaks] - offsetppm)
            idx = np.argmin(diff)
            offset1 = ppm_orig[candidate_peaks][idx]
        else:
            if interactive:
                plt.figure()
                plt.plot(ppm_orig, intensity1, 'k-')
                plt.plot(ppm_orig[candidate_peaks], intensity1[candidate_peaks], 'ro')
                plt.xlim(xlim)
                plt.gca().invert_xaxis()
                plt.title(f"Click near the reference peak for {sample1}")
                plt.xlabel("Chemical Shift (ppm)")
                plt.ylabel("Intensity")
                pts = plt.ginput(1, timeout=-1)
                plt.close()
                if pts:
                    click_x = pts[0][0]
                    diff = np.abs(ppm_orig[candidate_peaks] - click_x)
                    idx = np.argmin(diff)
                    offset1 = ppm_orig[candidate_peaks][idx]
                else:
                    print("No selection made; defaulting to no shift.")
                    offset1 = 0.0
            else:
                heights = intensity1[candidate_peaks]
                idx = np.argmax(heights)
                offset1 = ppm_orig[candidate_peaks][idx]
    offsets[sample1] = offset1
    new_axis = ppm_orig - offset1
    ref_intensity = {}
    for sample in sample_cols:
        intensity = df[sample].values
        thresh_val = thresh * np.max(intensity)
        peaks, _ = find_peaks(intensity, height=thresh_val)
        if len(peaks) == 0:
            print(f"No peaks found in {sample}; no referencing applied.")
            offset_i = 0.0
        else:
            candidate_ppms = ppm_orig[peaks]
            idx = np.argmin(np.abs(candidate_ppms))
            offset_i = candidate_ppms[idx]
        offsets[sample] = offset_i
        shifted_axis = ppm_orig - offset_i
        intensity_interp = np.interp(new_axis, shifted_axis, intensity)
        ref_intensity[sample] = intensity_interp
        print(f"Sample '{sample}' referenced using offset {offset_i:.4f} ppm.")
    referenced_df = pd.DataFrame({axis_col: new_axis})
    for sample in sample_cols:
        referenced_df[sample] = ref_intensity[sample]
    return referenced_df, offsets

def reference_to_target_ppm(
    df: pd.DataFrame,
    target_ppm: float = 0.0,
    search_window: tuple[float, float] = (-0.35, 0.35),  # window around target
    axis_col: str = "Chemical Shift (ppm)",
    use_abs: bool = True,   # use absolute peak height (handles phase issues)
    fill_edges: bool = True # fill NaNs created at the edges after shifting
) -> tuple[pd.DataFrame, dict]:
    """
    Shift *each* spectrum so that its most intense peak within [target+win_lo, target+win_hi]
    lands at `target_ppm`. Returns (shifted_df, offsets_dict), where offsets_dict maps sample->ppm_shift.
    Positive offset means spectrum is shifted to higher ppm.
    """
    if df is None or df.empty:
        return df, {}

    out = df.copy()
    ppm = out[axis_col].astype(float).values
    # Ensure ascending axis for np.interp
    if not np.all(np.diff(ppm) >= 0):
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
    # shift each sample column by interpolation
    for col in [c for c in out.columns if c != axis_col]:
        y = pd.to_numeric(out[col], errors="coerce").values
        y_win = y[mask]
        ppm_win = ppm[mask]
        if np.all(~np.isfinite(y_win)) or y_win.size == 0:
            # no valid data in window → no shift
            offsets[col] = 0.0
            continue

        # pick peak location in the window
        if use_abs:
            idx = int(np.nanargmax(np.abs(y_win)))
        else:
            idx = int(np.nanargmax(y_win))
        peak_ppm = float(ppm_win[idx])

        # amount we need to move this spectrum
        shift = float(target_ppm - peak_ppm)  # +shift → move to higher ppm
        offsets[col] = shift

        # build shifted intensities on the common ppm grid
        # new_y(x) = old_y(x - shift)
        src_x = ppm - shift
        new_y = np.interp(ppm, src_x, y, left=np.nan, right=np.nan)

        if fill_edges:
            s = pd.Series(new_y)
            new_y = s.fillna(method="bfill").fillna(method="ffill").values

        out[col] = new_y

    return out, offsets


def align_samples_using_icoshift(df, n_intervals=50, target='maxcorr'):
    ppm = df.iloc[:, 0].to_numpy()
    Xnp = df.iloc[:, 1:].to_numpy()
    if np.isnan(Xnp).all():
        raise ValueError("Spectral data contains only NaNs!")
    if ppm.shape[0] != Xnp.shape[0]:
        raise ValueError(f"Mismatch: ppm has {ppm.shape[0]} values, but Xnp has {Xnp.shape[0]} rows.")
    Xnp = Xnp.T
    Xnp = np.nan_to_num(Xnp)
    fix_int_size = Icoshift()
    fix_int_size.signals = Xnp
    fix_int_size.unit_vector = ppm
    fix_int_size.inter = ('n_intervals', n_intervals)
    fix_int_size.target = target
    fix_int_size.run()
    aligned_df = pd.DataFrame(fix_int_size.result.T, columns=df.columns[1:])
    aligned_df.insert(0, df.columns[0], ppm)
    return aligned_df

def RAFFT_df(data, reference_idx=0, shift_ppm=None, lookahead=1):
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values
    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T
    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")
    reference_spectrum = intensities[reference_idx, :]
    if shift_ppm is not None:
        dppm = np.abs(axis[1] - axis[0])
        shift = int(round(shift_ppm / dppm))
    else:
        shift = len(reference_spectrum)
    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = recur_align(intensities[i, :], reference_spectrum, shift, lookahead)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

# -------------------------- Helper Functions for RAFFT --------------------------
def recur_align(spectrum, reference, shift, lookahead):
    if len(spectrum) < 10:
        return spectrum
    lag = fft_corr(spectrum, reference, shift)
    if lag == 0 and lookahead <= 0:
        return spectrum
    else:
        if lag == 0:
            lookahead -= 1
        if abs(lag) < len(spectrum):
            aligned = move_seg(spectrum, lag)
        else:
            aligned = spectrum.copy()
        mid = find_mid(aligned)
        first_seg = recur_align(aligned[:mid], reference[:mid], shift, lookahead)
        second_seg = recur_align(aligned[mid:], reference[mid:], shift, lookahead)
        return np.concatenate([first_seg, second_seg])

def fft_corr(spectrum, target, shift):
    M = len(target)
    diff = 1e6
    for i in range(1, 21):
        curdiff = (2**i) - M
        if curdiff > 0 and curdiff < diff:
            diff = curdiff
    diff = int(diff)
    target_pad = np.pad(target, (0, diff), mode='constant')
    spectrum_pad = np.pad(spectrum, (0, diff), mode='constant')
    M_new = len(target_pad)
    X = np.fft.fft(target_pad)
    Y = np.fft.fft(spectrum_pad)
    R = (X * np.conjugate(Y)) / M_new
    rev = np.fft.ifft(R)
    vals = np.real(rev)
    maxi = -1
    maxpos = 0
    shift = min(shift, M_new)
    for i in range(shift):
        if vals[i] > maxi:
            maxi = vals[i]
            maxpos = i
        if vals[M_new - i - 1] > maxi:
            maxi = vals[M_new - i - 1]
            maxpos = M_new - i - 1
    if maxi < 0.1:
        return 0
    if maxpos > len(vals) / 2:
        lag = maxpos - len(vals) - 1
    else:
        lag = maxpos - 1
    return lag

def move_seg(seg, lag):
    if lag == 0 or lag >= len(seg):
        return seg
    if lag > 0:
        ins = np.full(lag, seg[0])
        return np.concatenate([ins, seg[:-lag]])
    else:
        lag_abs = abs(lag)
        ins = np.full(lag_abs, seg[-1])
        return np.concatenate([seg[lag_abs:], ins])

def find_mid(spec):
    M = int(np.ceil(len(spec) / 2))
    offset = int(np.floor(M / 4))
    start = max(M - offset, 0)
    end = min(M + offset, len(spec))
    spec_segment = spec[start:end]
    I = np.argmin(spec_segment)
    mid = I + start
    return mid

def PAFFT_df(data, segSize_ppm, reference_idx=0, shift_ppm=None):
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values
    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T
    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")
    reference_spectrum = intensities[reference_idx, :]
    dppm = np.abs(axis[1] - axis[0])
    if shift_ppm is not None:
        shift = int(round(shift_ppm / dppm))
    else:
        shift = len(reference_spectrum)
    segSize = int(round(segSize_ppm / dppm))
    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = PAFFT(intensities[i, :], reference_spectrum, segSize, shift)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

def PAFFT(spectrum, reference, segSize, shift):
    n_points = len(spectrum)
    aligned_segments = []
    startpos = 0
    while startpos < n_points:
        endpos = startpos + segSize * 2
        if endpos >= n_points:
            samseg = spectrum[startpos:]
            refseg = reference[startpos:]
        else:
            samseg = spectrum[startpos + segSize: endpos - 1]
            refseg = reference[startpos + segSize: endpos - 1]
            minpos = find_min(samseg, refseg)
            endpos = startpos + minpos + segSize
            samseg = spectrum[startpos:endpos]
            refseg = reference[startpos:endpos]
        lag = fft_corr(samseg, refseg, shift)
        moved = move_seg(samseg, lag)
        aligned_segments.append(moved)
        startpos = endpos + 1
    aligned_full = np.concatenate(aligned_segments)
    if len(aligned_full) < n_points:
        aligned_full = np.pad(aligned_full, (0, n_points - len(aligned_full)), mode='edge')
    else:
        aligned_full = aligned_full[:n_points]
    return aligned_full

def find_min(samseg, refseg):
    Cs = np.sort(samseg)
    Is = np.argsort(samseg)
    Cr = np.sort(refseg)
    Ir = np.argsort(refseg)
    n_limit = max(1, int(round(len(Cs) / 20)))
    for i in range(n_limit):
        for j in range(n_limit):
            if Ir[j] == Is[i]:
                return Is[i]
    return Is[0]

def apply_alignment_by_regions(aligned_df, region_alignments):
    aligned_parts = []
    for region_info in region_alignments:
        start_ppm, end_ppm = region_info['region']
        align_func = region_info['align_func']
        params = region_info.get('params', {})
        region_mask = (aligned_df["Chemical Shift (ppm)"] >= start_ppm) & (aligned_df["Chemical Shift (ppm)"] <= end_ppm)
        region_df = aligned_df.loc[region_mask].copy()
        aligned_region = align_func(region_df, **params)
        aligned_parts.append(aligned_region)
    final_aligned_df = pd.concat(aligned_parts, ignore_index=True)
    final_aligned_df = final_aligned_df.sort_values("Chemical Shift (ppm)").reset_index(drop=True)
    return final_aligned_df

# --------------------------------------------------------------------------
#               5) EXPORT FUNCTIONS
# --------------------------------------------------------------------------
def sanitize_string(s):
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def export_metaboanalyst(aligned_df, df_metadata,
                         sample_id_col="NMR_filename",
                         class_col="ATTRIBUTE_classification",
                         output_file="metaboanalyst_input.csv"):
    orig_col_names = list(aligned_df.columns)
    sanitized_sample_cols = [sanitize_string(s) for s in orig_col_names[1:]]
    sanitized_col_names = [orig_col_names[0]] + sanitized_sample_cols
    aligned_df = aligned_df.copy()
    aligned_df.columns = sanitized_col_names
    sample_cols = sanitized_col_names[1:]
    meta = df_metadata.copy()
    meta[sample_id_col] = meta[sample_id_col].apply(sanitize_string)
    meta[class_col] = meta[class_col].apply(sanitize_string)
    meta_indexed = meta.set_index(sample_id_col)
    classification_series = meta_indexed.reindex(sample_cols)[class_col]
    valid_sample_cols = classification_series.dropna().index.tolist()
    if len(valid_sample_cols) < len(sample_cols):
        missing = set(sample_cols) - set(valid_sample_cols)
        print(f"Warning: The following sample IDs are missing metadata and will be excluded: {missing}")
    classification_row = [""]
    classification_row.extend(classification_series.loc[valid_sample_cols].values)
    new_df = aligned_df[['Chemical Shift (ppm)'] + valid_sample_cols].copy()
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_df.columns.tolist())
        writer.writerow(classification_row)
        for i in range(len(new_df)):
            writer.writerow(new_df.iloc[i].values)
    print(f"MetaboAnalyst input CSV saved as: {output_file}")
    return new_df

# --------------------------------------------------------------------------
#               6) QUALITY CONTROL CHECK
# --------------------------------------------------------------------------
def calculate_peak_area(df, peak_intervals, x_axis='Chemical Shift (ppm)'):
    """
    Calculate the area under the curve for specified peak intervals for each sample.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the spectral data. Must have a column with chemical shift values 
        (e.g., 'Chemical Shift (ppm)') and the remaining columns are sample intensities.
    peak_intervals : list of tuples
        List of tuples, where each tuple defines the start and end of the peak interval (in ppm).
        Example: [(7.79, 7.83), (7.5, 7.55)]
    x_axis : str
        Name of the column containing chemical shift values.
    
    Returns:
    --------
    areas_dict : dict
        A dictionary where keys are string representations of the peak intervals and values are 
        pandas Series with sample names as index and calculated areas as values.
    """
    areas_dict = {}
    # Get the chemical shift vector
    x = df[x_axis].values
    
    for (start, end) in peak_intervals:
        # Create a boolean mask for the selected interval.
        mask = (x >= start) & (x <= end)
        area_values = {}
        # For each sample column (skip the x-axis column)
        for col in df.columns:
            if col == x_axis:
                continue
            y = df[col].values
            # Restrict both x and y to the interval
            x_segment = x[mask]
            y_segment = y[mask]
            if len(x_segment) > 1:
                # Use the trapezoidal rule to compute the area
                area = np.trapz(y_segment, x_segment)
            else:
                area = np.nan
            area_values[col] = area
        areas_dict[f"[{start}, {end}]"] = pd.Series(area_values)
    
    return areas_dict

def plot_peak_areas(areas_dict, output_dir="images"):
    """
    Plot scatter plots for each peak interval showing the area under the curve for each sample.
    Overlays a red dashed line for the mean and a red shaded region for one standard deviation.
    Each plot is saved in the specified output directory.
    
    Parameters:
    -----------
    areas_dict : dict
        Dictionary with keys as peak interval strings and values as pandas Series with areas.
    output_dir : str
        Directory where the plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for region, series in areas_dict.items():
        plt.figure(figsize=(10, 6))
        x_positions = np.arange(len(series))
        plt.scatter(x_positions, series.values, color='blue', s=50, label="Area")
        
        # Calculate mean and standard deviation
        mean_val = series.mean()
        std_val = series.std()
        
        # Plot mean line and standard deviation band
        plt.axhline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")
        plt.fill_between(x_positions, mean_val - std_val, mean_val + std_val, color='red', alpha=0.2, label=f"Std = {std_val:.2f}")
        
        plt.xticks(x_positions, series.index, rotation=90)
        plt.xlabel("Sample")
        plt.ylabel("Area Under Curve")
        plt.title(f"Peak Area for {region} ppm")
        plt.legend()
        plt.tight_layout()
        
        # Save the figure to the output directory
        filename = f"PeakArea_{region.replace('[', '').replace(']', '').replace(', ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()
        plt.close()

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def integrate_area_html_all(aligned_df, peaks, output_folder="html_images", base_filename="integration_plot"):
    """
    Computes the area under the curve (AUC) for each sample in the given DataFrame within
    specified retention time (RT) intervals defined in the peaks list. For each sample, an
    interactive Plotly HTML plot is created that highlights the integration regions.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame where the first column is "RT (min)" (retention time) and the remaining columns 
        are the intensities for each sample.
    peaks : list
        List of retention time boundaries in the form 
            [lower_rt1, upper_rt1, lower_rt2, upper_rt2, ...].
        The length of peaks must be even.
    output_folder : str, optional
        Folder (inside the "images" directory) in which to save the HTML plots (default "html_images").
    base_filename : str, optional
        Base name for the output files. Each file will be named as 
        "{base_filename}_{sample_name}.html" (default "integration_plot").
    
    Returns
    -------
    auc_df : pd.DataFrame
        A DataFrame where rows are samples and columns are each integration region 
        (e.g., "Region 1", "Region 2", ...) containing the corresponding AUC values.
    """
    # Check that peaks has an even number of elements.
    if len(peaks) % 2 != 0:
        raise ValueError("The peaks list must contain an even number of elements (pairs of lower and upper limits).")
    
    # Create the base output directory and subfolder for HTML plots.
    base_output_dir = "images"
    html_output_dir = os.path.join(base_output_dir, output_folder)
    os.makedirs(html_output_dir, exist_ok=True)
    
    # Extract retention time (RT) from the first column and convert to numeric values
    rt = pd.to_numeric(aligned_df.iloc[:, 0], errors='coerce').values
    
    sample_names = aligned_df.columns[1:]
    num_regions = len(peaks) // 2
    
    # Prepare a dictionary to collect AUC values per sample and per region.
    auc_data = {f"Region {i+1}": [] for i in range(num_regions)}
    auc_data["Sample"] = []
    
    # Loop over each sample to compute integration and create a plot
    for sample in sample_names:
        intensities = pd.to_numeric(aligned_df[sample], errors='coerce').values
        
        # For storing region-specific AUC values for this sample.
        sample_auc = []
        
        # Create the Plotly figure for this sample
        fig = go.Figure()
        
        # Add the full spectrum trace for the sample
        fig.add_trace(go.Scatter(
            x=rt,
            y=intensities,
            mode='lines',
            name=sample,
            line=dict(color='black')
        ))
        
        # Loop through each integration region (pair of limits)
        for i in range(0, len(peaks), 2):
            lower_rt = peaks[i]
            upper_rt = peaks[i+1]
            mask = (rt >= lower_rt) & (rt <= upper_rt)
            if np.sum(mask) >= 2:
                auc = np.trapz(intensities[mask], rt[mask])
            else:
                auc = 0  # or np.nan if preferred
            sample_auc.append(auc)
            
            # Add trace for the integration region
            fig.add_trace(go.Scatter(
                x=rt[mask],
                y=intensities[mask],
                mode='lines',
                name=f"Region {i//2 + 1}",
                line=dict(color='skyblue'),
                fill='tozeroy',
                opacity=0.5
            ))
            
            # Add vertical dashed lines for region boundaries
            fig.add_shape(
                type="line",
                x0=lower_rt,
                x1=lower_rt,
                y0=min(intensities),
                y1=max(intensities),
                line=dict(color="red", dash="dash")
            )
            fig.add_shape(
                type="line",
                x0=upper_rt,
                x1=upper_rt,
                y0=min(intensities),
                y1=max(intensities),
                line=dict(color="red", dash="dash")
            )
        
        # Store the sample name and its region-specific AUC values
        auc_data["Sample"].append(sample)
        for idx, auc_val in enumerate(sample_auc):
            auc_data[f"Region {idx+1}"].append(auc_val)
        
        # Update layout for the current sample plot
        fig.update_layout(
            title=f"Integration Regions for Sample: {sample}",
            xaxis_title="RT (min)",
            yaxis_title="Intensity",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save the figure as an interactive HTML file in the designated folder
        output_filename = f"{base_filename}_{sample}.html"
        output_path = os.path.join(html_output_dir, output_filename)
        fig.write_html(output_path)
        print(f"Integration plot for sample '{sample}' saved as: {output_path}")
    
    # Create a DataFrame from the auc_data dictionary
    auc_df = pd.DataFrame(auc_data).set_index("Sample")
    
    return auc_df



import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_overlayed_spectra(aligned_df, peaks, output_filename="overlayed_chromatograms.html", output_folder="images"):
    """
    Creates an interactive Plotly HTML plot that overlays the chromatograms for all samples 
    (with retention time in the first column and intensities in subsequent columns) and 
    highlights the integration regions defined by the peaks list.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame where the first column is "RT (min)" (retention time) and the remaining columns 
        are the intensities for each sample.
    peaks : list
        List of retention time boundaries in the form 
            [lower_rt1, upper_rt1, lower_rt2, upper_rt2, ...].
        The length of peaks must be even.
    output_filename : str, optional
        Name of the output HTML file (default "overlayed_chromatograms.html").
    output_folder : str, optional
        Folder (inside the "images" directory) in which to save the HTML plot (default "html_images_overlay").
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure with overlayed chromatograms and integrated regions.
    """
    # Check that peaks has an even number of elements.
    if len(peaks) % 2 != 0:
        raise ValueError("The peaks list must contain an even number of elements (pairs of lower and upper limits).")
    
    # Create the output directory for HTML plots
    base_output_dir = "images"
    html_output_dir = os.path.join(base_output_dir, output_folder)
    os.makedirs(html_output_dir, exist_ok=True)
    
    # Extract retention time (RT) from the first column and convert to numeric values
    rt = pd.to_numeric(aligned_df.iloc[:, 0], errors='coerce').values
    
    # Initialize the Plotly figure
    fig = go.Figure()
    
    # Overlay each sample's chromatogram
    sample_names = aligned_df.columns[1:]
    for sample in sample_names:
        intensities = pd.to_numeric(aligned_df[sample], errors='coerce').values
        fig.add_trace(go.Scatter(
            x=rt,
            y=intensities,
            mode='lines',
            name=sample,
            line=dict(width=1),
            opacity=0.7
        ))
    
    # Add shapes (shaded regions and vertical lines) for each integration region
    num_regions = len(peaks) // 2
    for i in range(num_regions):
        lower_rt = peaks[2*i]
        upper_rt = peaks[2*i + 1]
        # Add a shaded region
        fig.add_vrect(
            x0=lower_rt, x1=upper_rt,
            fillcolor="skyblue", opacity=0.3, line_width=0,
            annotation_text=f"Region {i+1}", annotation_position="top left"
        )
        # Optionally, add vertical dashed lines at the boundaries
        fig.add_shape(
            type="line",
            x0=lower_rt, x1=lower_rt,
            y0=min(rt), y1=max(rt),
            line=dict(color="red", dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=upper_rt, x1=upper_rt,
            y0=min(rt), y1=max(rt),
            line=dict(color="red", dash="dash")
        )
    
    # Update layout settings for a clear presentation
    fig.update_layout(
        title="Overlayed Chromatograms with Integration Regions",
        xaxis_title="Chemical Shift (ppm)",
        yaxis_title="Intensity",
        xaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save the interactive plot as an HTML file in the designated folder
    output_path = os.path.join(html_output_dir, output_filename)
    fig.write_html(output_path)
    print(f"Overlayed chromatograms plot saved as: {output_path}")
    
    return fig


        
# --------------------------------------------------------------------------
#               7) CENTERING, NORMALIZATION & SCALING FUNCTIONS
# --------------------------------------------------------------------------
# Centering Functions
def log_transform(df, constant=1):
    df = df.copy()
    for column in df.columns:
        if column != "Chemical Shift (ppm)":
            adjusted = df[column] + constant
            adjusted = adjusted.where(adjusted > 0, np.finfo(float).eps)
            df[column] = np.log10(adjusted)
    return df

def sqrt_transform(df):
    df = df.copy()
    for column in df.columns:
        if column != "Chemical Shift (ppm)":
            df[column] = np.sqrt(df[column].clip(lower=0))
    return df

def cbrt_transform(df):
    df = df.copy()
    for column in df.columns:
        if column != "Chemical Shift (ppm)":
            df[column] = np.cbrt(df[column])
    return df

# Normalization Functions
def min_max_normalize(df):
    df = df.copy()
    for column in df.columns:
        if column != "Chemical Shift (ppm)":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def z_score_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'Chemical Shift (ppm)':
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
            else:
                df[column] = df[column] - mean_val
    return df

def normalize_by_control(df, control_column, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    control = df[control_column]
    for column in df.columns:
        if column != control_column and column not in exclude_columns and column != 'Chemical Shift (ppm)':
            df[column] = df[column] / control
    return df

def pqn_normalize(df, reference=None, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    cols = [col for col in df.columns if col != 'Chemical Shift (ppm)' and col not in exclude_columns]
    df_numeric = df[cols]
    if reference is None:
        reference = df_numeric.median(axis=1)
    quotients = df_numeric.divide(reference, axis=0)
    median_quotients = quotients.median(axis=0)
    df_norm = df_numeric.divide(median_quotients, axis=1)
    for col in df.columns:
        if col not in df_norm.columns:
            df_norm[col] = df[col]
    df_norm = df_norm[df.columns]
    return df_norm

def std_dev_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'Chemical Shift (ppm)':
            std_val = df[column].std()
            if std_val != 0:
                df[column] = df[column] / std_val
    return df

def median_normalize(df, target_median=1.0, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'Chemical Shift (ppm)':
            median_val = df[column].median()
            if median_val != 0:
                df[column] = (df[column] / median_val) * target_median
    return df

def quantile_normalize(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    else:
        exclude_columns = list(set(exclude_columns + ["RT(min)", "Chemical Shift (ppm)"]))
    norm_cols = [col for col in df.columns if col not in exclude_columns]
    df_numeric = df[norm_cols]
    sorted_df = pd.DataFrame(
        np.sort(df_numeric.values, axis=0),
        index=df_numeric.index,
        columns=df_numeric.columns
    )
    rank_means = sorted_df.mean(axis=1)
    df_normalized = df_numeric.copy()
    for col in df_numeric.columns:
        order = df_numeric[col].argsort()
        normalized_vals = pd.Series(rank_means.values, index=order)
        df_normalized[col] = normalized_vals.sort_index().values
    for col in exclude_columns:
        df_normalized[col] = df[col]
    df_normalized = df_normalized[df.columns]
    return df_normalized

# Scaling Functions
def min_max_scale(df, new_min=0, new_max=1, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val != min_val:
                df[column] = (df[column] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return df

def standard_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
            else:
                df[column] = df[column] - mean_val
    return df

def robust_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            median_val = df[column].median()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR != 0:
                df[column] = (df[column] - median_val) / IQR
    return df

def mean_center(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            df[column] = df[column] - df[column].mean()
    return df

def auto_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - df[column].mean()) / std_val
            else:
                df[column] = df[column] - df[column].mean()
    return df

from sklearn.preprocessing import StandardScaler

def auto_scale_m(df, exclude_columns=None):
    """
    Applies auto-scaling (mean-centering and division by std) to all columns except those excluded.
    This replicates scikit-learn's StandardScaler approach, commonly used in MetaboAnalyst.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric columns to be scaled.
    exclude_columns : list or None
        Columns that should not be scaled (e.g., "Chemical Shift (ppm)", "RT(min)").

    Returns
    -------
    df_scaled : pd.DataFrame
        A new DataFrame with scaled columns (except excluded ones).
    """
    df_scaled = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]

    # Identify columns to be scaled
    scale_cols = [col for col in df_scaled.columns if col not in exclude_columns]

    # Apply StandardScaler to those columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_scaled[scale_cols])

    # Replace the original columns with scaled values
    df_scaled[scale_cols] = scaled_data

    return df_scaled

def pareto_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - df[column].mean()) / np.sqrt(std_val)
            else:
                df[column] = df[column] - df[column].mean()
    return df

def range_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            rng = df[column].max() - df[column].min()
            if rng != 0:
                df[column] = (df[column] - df[column].mean()) / rng
    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_normalization_plots(
    before_df, 
    after_df, 
    sample_limit=50, 
    exclude_columns=None, 
    title_before='Before Normalization',
    title_after='After Normalization',
    show_full_xrange=False,
    zoom_in_std=None
):
    """
    Creates a 2x2 grid of plots:
      - Top-left: Density plot of all values before normalization
      - Top-right: Density plot of all values after normalization 
                   (range controlled by show_full_xrange or zoom_in_std).
      - Bottom-left: Box plot (at most sample_limit columns) before normalization
      - Bottom-right: Box plot (at most sample_limit columns) after normalization

    Parameters
    ----------
    before_df : pd.DataFrame
        DataFrame containing the data prior to normalization.
    after_df : pd.DataFrame
        DataFrame containing the data after normalization.
    sample_limit : int
        Maximum number of columns to display in the box plots.
    exclude_columns : list or None
        List of columns to exclude from both DataFrames (e.g. axis columns like "Chemical Shift (ppm)").
    title_before : str
        Title for the "before" plots.
    title_after : str
        Title for the "after" plots.
    show_full_xrange : bool
        If True, the top-right density plot will show the entire range of values.
        If False, it will restrict the view to the 1st and 99th percentiles (unless zoom_in_std is used).
    zoom_in_std : float or None
        If not None, the top-right density plot will show mean ± (zoom_in_std * standard deviation).
        This overrides the percentile-based or full-range display.
    """
    # Copy DataFrames so we don't modify the originals
    df_before = before_df.copy()
    df_after = after_df.copy()
    
    # Exclude certain columns if requested
    if exclude_columns is not None:
        df_before.drop(columns=exclude_columns, errors='ignore', inplace=True)
        df_after.drop(columns=exclude_columns, errors='ignore', inplace=True)
    
    # Flatten the entire DataFrame values for density plots
    all_values_before = df_before.values.flatten()
    all_values_after = df_after.values.flatten()
    
    # Calculate some statistics for controlling the x-axis range
    lower_limit = np.percentile(all_values_after, 1)
    upper_limit = np.percentile(all_values_after, 99)
    mean_val = np.mean(all_values_after)
    std_val = np.std(all_values_after)

    # Limit columns for box plot (for readability)
    limited_cols_before = df_before.columns[:sample_limit]
    limited_cols_after = df_after.columns[:sample_limit]
    
    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # --- Top-left: Density of all values BEFORE normalization ---
    sns.kdeplot(x=all_values_before, ax=axes[0,0], color='steelblue', fill=True)
    axes[0,0].set_title(title_before, fontsize=12)
    axes[0,0].set_xlabel("Value")
    axes[0,0].set_ylabel("Density")
    
    # --- Top-right: Density of all values AFTER normalization ---
    sns.kdeplot(x=all_values_after, ax=axes[0,1], color='darkorange', fill=True)
    axes[0,1].set_title(title_after, fontsize=12)
    axes[0,1].set_xlabel("Value")
    axes[0,1].set_ylabel("Density")
    
    # Control the x-axis range for the "after" density plot
    if zoom_in_std is not None:
        # Show mean ± zoom_in_std * std_val
        axes[0,1].set_xlim(mean_val - zoom_in_std * std_val, 
                           mean_val + zoom_in_std * std_val)
    else:
        # If zoom_in_std is not used, check if we should show full range or restrict to 1st-99th percentiles
        if not show_full_xrange:
            axes[0,1].set_xlim(lower_limit, upper_limit)
        # else: do nothing => show the full range automatically

    # --- Bottom-left: Box plot BEFORE normalization ---
    sns.boxplot(data=df_before[limited_cols_before], ax=axes[1,0], color='steelblue')
    axes[1,0].tick_params(axis='x', rotation=90)
    axes[1,0].set_title(title_before, fontsize=12)
    axes[1,0].set_ylabel("Value")
    
    # --- Bottom-right: Box plot AFTER normalization ---
    sns.boxplot(data=df_after[limited_cols_after], ax=axes[1,1], color='darkorange')
    axes[1,1].tick_params(axis='x', rotation=90)
    axes[1,1].set_title(title_after, fontsize=12)
    axes[1,1].set_ylabel("Value")
    
    # Show the final figure
    plt.show()



# --------------------------------------------------------------------------
#               8) PCA & PLS-DA, VIP FUNCTIONS
# --------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import plotly.express as px
from types import SimpleNamespace  # Import for converting dict to object

def nipals_pca(X, n_components, thresh=1e-15):
    """
    Perform PCA using the iterative NIPALS algorithm.
    
    Parameters:
        X : np.ndarray
            Data matrix with shape (n_samples, n_features).
        n_components : int
            Number of principal components to compute.
        thresh : float, optional
            Convergence threshold for the iterative algorithm.
    
    Returns:
        T : np.ndarray
            Scores matrix with shape (n_samples, n_components).
        P : np.ndarray
            Loadings matrix with shape (n_components, n_features).
        variance : np.ndarray
            Explained variance (as fraction of total variance) for each component.
    """
    n_samples, n_features = X.shape
    T = np.zeros((n_samples, n_components))  # scores
    P = np.zeros((n_features, n_components)) # loadings (each column is a loading vector)
    variance = np.zeros(n_components)        # variance explained by each component
    
    Xi = X.copy()  # working copy of X
    total_variance = np.sum(X ** 2)
    
    for i in range(n_components):
        residual = 1.0
        p_initial = np.zeros(n_features)
        # Initialize t as the first column of Xi.
        t = Xi[:, 0].copy()
        
        while residual > thresh:
            # Compute loading: project Xi onto t.
            p = np.dot(Xi.T, t) / np.dot(t, t)
            # Normalize p to have unit length.
            norm_p = np.sqrt(np.dot(p, p))
            if norm_p != 0:
                p = p / norm_p
            # Recalculate score: project Xi onto p.
            t = np.dot(Xi, p) / np.dot(p, p)
            # Check convergence based on change in loadings.
            E = p_initial - p
            residual = np.dot(E, E)
            p_initial = p.copy()
        
        T[:, i] = t
        P[:, i] = p
        # Remove the contribution of the extracted component.
        Xi = Xi - np.outer(t, p)
        # Calculate the proportion of variance explained by this component.
        variance[i] = np.sum((np.outer(t, p))**2) / total_variance
    
    # Transpose loadings so that each row corresponds to a component.
    return T, P.T, variance

def perform_pca_analysis(data, pc_x=1, pc_y=2, n_components=None, variance_threshold=90,
                          metadata=None, color_column="ATTRIBUTE_group", sample_id_col="NMR_filename",
                          output_dir='images', score_plot_filename=None, ev_plot_filename=None,
                          show_fig=True):
    """
    Performs PCA using an iterative NIPALS algorithm.
    The function computes PCA scores, loadings, explained variance, and creates interactive plots.
    
    Parameters:
        data : pd.DataFrame or np.ndarray
            Data with features as rows and samples as columns. The function transposes it so that
            each row represents a sample.
        pc_x, pc_y : int
            Principal components to plot on the x and y axes.
        n_components : int, optional
            Number of principal components to compute. If None, the number is determined based on
            the variance_threshold.
        variance_threshold : float, optional
            Minimum cumulative explained variance (in %) to determine the number of components.
        metadata : pd.DataFrame, optional
            Metadata to merge with the PCA scores DataFrame.
        color_column : str
            Column in metadata used for coloring the score plot.
        sample_id_col : str
            Column name for sample IDs in the metadata.
        output_dir : str
            Directory to save plot HTML files.
        score_plot_filename, ev_plot_filename : str, optional
            Filenames for saving the score and explained variance plots.
        show_fig : bool
            Whether to display the plots.
    
    Returns:
        tuple: (pca_model, scores_df, explained_variance) where:
            - pca_model is an object with attributes 'scores', 'loadings', 'variance', and 'components_'.
            - scores_df is a DataFrame with PCA scores (and merged metadata if provided).
            - explained_variance is an array with percentage explained variance per component.
    """
    # If data is a DataFrame, drop the "Chemical Shift (ppm)" column if it exists.
    if isinstance(data, pd.DataFrame):
        if "Chemical Shift (ppm)" in data.columns:
            data = data.drop(columns=["Chemical Shift (ppm)"])

    # Transpose data so that rows represent samples.
    if isinstance(data, pd.DataFrame):
        X = data.transpose().values
        sample_ids = data.transpose().index
    else:
        X = data.T
        sample_ids = np.arange(X.shape[0])
    
    # *** Center the data to mimic MATLAB's PCA centering ***
    mu = np.mean(X, axis=0)
    X = X - mu

    # Determine the maximum possible number of components.
    max_comp = min(X.shape)
    
    # First, run NIPALS for the maximum number of components to calculate cumulative variance.
    T_full, P_full, var_full = nipals_pca(X, n_components=max_comp)
    cum_var = np.cumsum(var_full) * 100  # cumulative explained variance in %
    
    # Determine the number of components needed to reach the variance threshold.
    n_comp = np.argmax(cum_var >= variance_threshold) + 1
    # Ensure at least the plotted components are computed.
    n_comp = max(n_comp, pc_x, pc_y)
    
    # Override if user provided a fixed n_components.
    if n_components is not None:
        n_comp = n_components
    
    # Run NIPALS again with the desired number of components.
    scores, loadings, variance = nipals_pca(X, n_components=n_comp)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    
    # Create a DataFrame for PCA scores.
    scores_df = pd.DataFrame(scores, columns=comp_labels, index=sample_ids)
    
    # Merge with metadata if provided.
    if metadata is not None:
        scores_df = scores_df.merge(metadata[[sample_id_col, color_column]],
                                    left_index=True, right_on=sample_id_col, how='left')
    
    # Convert explained variance fractions to percentages.
    explained_variance = np.array(variance) * 100
    
    # Create axis labels that include the explained variance.
    xlabel = f"PC{pc_x} ({explained_variance[pc_x-1]:.1f}% explained variance)"
    ylabel = f"PC{pc_y} ({explained_variance[pc_y-1]:.1f}% explained variance)"
    
    # Create the PCA score scatter plot.
    fig_scores = px.scatter(
        scores_df, 
        x=f"PC{pc_x}", 
        y=f"PC{pc_y}",
        color=color_column if metadata is not None else None,
        title="PCA Score Plot",
        labels={
            f"PC{pc_x}": xlabel,
            f"PC{pc_y}": ylabel
        }
    )
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        score_plot_file = os.path.join(output_dir, score_plot_filename)
        fig_scores.write_html(score_plot_file)
    if show_fig:
        fig_scores.show()
    
    # Create explained variance bar plot.
    ev_df = pd.DataFrame({
        "Component": comp_labels, 
        "Explained Variance (%)": explained_variance
    })
    fig_ev = px.bar(
        ev_df, 
        x="Component", 
        y="Explained Variance (%)",
        title="Explained Variance by Principal Components",
        text=ev_df["Explained Variance (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_ev.update_traces(textposition='outside')
    if ev_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        ev_plot_file = os.path.join(output_dir, ev_plot_filename)
        fig_ev.write_html(ev_plot_file)
    if show_fig:
        fig_ev.show()
    
    # Package the NIPALS results in a dictionary as the PCA model.
    model_dict = {
        "scores": scores,
        "loadings": loadings,
        "variance": variance,
        "components_": loadings
    }
    
    # Convert the dictionary to an object with attribute access.
    pca_model = SimpleNamespace(**model_dict)
    
    return pca_model, scores_df, explained_variance


def plot_pca_loadings(data, pca_model, PC_choose=1, x_axis_col='Chemical Shift (ppm)', 
                      output_dir='images', output_file=None, save_fig=True, show_fig=True):
    """
    Plots the PCA loadings for a selected principal component.
    
    Parameters:
        data (DataFrame): Input data containing the x-axis values.
        pca_model (PCA object): Fitted PCA model with computed components.
        PC_choose (int): The principal component to plot (1-indexed).
        x_axis_col (str): Column in data for x-axis values.
        output_dir (str): Directory to save the plot.
        output_file (str): Filename for saving the plot.
        save_fig (bool): Whether to save the plot as an HTML file.
        show_fig (bool): Whether to display the plot.
    
    Returns:
        Plotly Figure object.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings.html'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        name=f'PC{PC_choose} Loading'
    ))
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component',
        xaxis=dict(autorange='reversed')
    )
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def plot_pca_scores(scores_df, pc_x, pc_y, explained_variance):
    fig = px.scatter(scores_df, x=f'PC{pc_x}', y=f'PC{pc_y}', text=scores_df.index, title=f'PCA Score Plot: PC{pc_x} vs PC{pc_y}')
    fig.update_layout(
        xaxis_title=f'PC{pc_x} ({explained_variance[pc_x-1]:.2f}%)',
        yaxis_title=f'PC{pc_y} ({explained_variance[pc_y-1]:.2f}%)'
    )
    fig.update_traces(marker=dict(size=7),
                      selector=dict(mode='markers+text'))
    fig.show()

    
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder

def perform_pls_da(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="ATTRIBUTE_localsampleid",
    n_components=2,
    output_dir="images",
    score_plot_filename=None,
    show_fig=True
):
    """
    Perform PLS-DA (Partial Least Squares Discriminant Analysis) by:
      1. One-hot encoding the class labels.
      2. Fitting a PLSRegression model.
      3. Plotting the first two latent variable scores, colored by group.

    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples (and rows as features).
        (If the chemical shift axis is included as a row or column, it will be ignored.)
    metadata : pd.DataFrame
        DataFrame containing sample metadata. Must include the sample_id_col (matching data columns)
        and a grouping column (group_col) for classification.
    group_col : str
        Column in metadata that contains the class/group labels (e.g., "ATTRIBUTE_group").
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns (e.g., "ATTRIBUTE_localsampleid").
    n_components : int
        Number of latent variables (components) to use in the PLS model. Default=2 for a 2D plot.
    output_dir : str
        Directory where the HTML plot will be saved if score_plot_filename is not None.
    score_plot_filename : str or None
        If provided, the interactive plot is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly figure is displayed.

    Returns
    -------
    pls_model : PLSRegression
        The fitted PLS regression model.
    scores_df : pd.DataFrame
        DataFrame containing the latent variable scores for each sample, merged with group labels.
    """
    # If data is a DataFrame, remove any row or column named "Chemical Shift (ppm)".
    if isinstance(data, pd.DataFrame):
        if "Chemical Shift (ppm)" in data.columns:
            data = data.drop(columns=["Chemical Shift (ppm)"])
        if "Chemical Shift (ppm)" in data.index:
            data = data.drop(index=["Chemical Shift (ppm)"])
    
    # Ensure output directory exists if we plan to save
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Transpose data so rows = samples, columns = features
    X = data.transpose()  # shape = [n_samples, n_features]
    
    # 2. Align the metadata so that it matches the rows of X
    sample_index_df = pd.DataFrame({sample_id_col: X.index})
    merged_df = sample_index_df.merge(metadata, on=sample_id_col, how='left')
    
    # 3. One-hot encode the group labels for PLS-DA
    groups = merged_df[group_col].astype(str).values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(groups)  # shape = [n_samples, n_classes]
    
    # 4. Fit the PLS regression model
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, Y)
    
    # 5. Extract x-scores (latent variable scores for each sample)
    x_scores = pls_model.x_scores_
    lv_cols = [f"LV{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(x_scores, columns=lv_cols, index=X.index)
    
    # (A) Reset index and rename so sample IDs become a column named sample_id_col
    scores_df = scores_df.reset_index().rename(columns={"index": sample_id_col})
    
    # (B) Merge on the sample_id_col to get group labels
    scores_df = scores_df.merge(
        merged_df[[sample_id_col, group_col]],
        on=sample_id_col,
        how='left'
    )
    
    # 6. Plot the first two latent variables in an interactive scatter plot
    if n_components >= 2:
        fig = px.scatter(
            scores_df,
            x="LV1",
            y="LV2",
            color=group_col,
            hover_data=[sample_id_col],
            title="PLS-DA Score Plot (LV1 vs. LV2)",
            labels={"LV1": "Latent Variable 1", "LV2": "Latent Variable 2"}
        )
        if score_plot_filename is not None:
            html_path = os.path.join(output_dir, score_plot_filename)
            fig.write_html(html_path)
        if show_fig:
            fig.show()
    else:
        print(f"n_components={n_components} < 2, so no 2D score plot was generated.")
    
    return pls_model, scores_df


    
    
def plot_plsda_loadings(data, plsr_model, component=1, x_axis_col='RT(min)', 
                        output_dir='images', output_file=None, save_fig=True, show_fig=True):
    loadings = plsr_model.x_loadings_
    if component < 1 or component > loadings.shape[1]:
        raise ValueError(f"Component number must be between 1 and {loadings.shape[1]}, got {component}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings[:, component - 1],
        mode='lines',
        name=f'PLS-DA Component {component} Loading'
    ))
    fig.update_layout(
        title=f'PLS-DA Component {component} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value'
    )
    if save_fig:
        if output_file is None:
            output_file = f'PLSDA_Component_{component}_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def calculate_vip_scores(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip

def analyze_vip_scores(pls_model, X, top_n=10,
                       save_df=False, output_dir='images', output_file_df=None,
                       plot_fig=False, save_fig=False, output_file_plot=None, show_fig=True):
    vip = calculate_vip_scores(pls_model, X)
    vip_df = pd.DataFrame({'Variable': X.columns, 'VIP Score': vip})
    vip_df = vip_df.sort_values(by='VIP Score', ascending=False)
    top_vip = vip_df.head(top_n)
    if save_df:
        if output_file_df is None:
            output_file_df = 'VIP_scores.csv'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file_df)
        top_vip.to_csv(output_path, index=False)
        print(f"VIP scores saved as: {output_path}")
    if plot_fig:
        fig = px.bar(top_vip, x='Variable', y='VIP Score',
                     title='Top VIP Scores',
                     text='VIP Score')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        if save_fig:
            if output_file_plot is None:
                output_file_plot = 'VIP_scores_plot.html'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, output_file_plot)
            fig.write_html(plot_path)
            print(f"VIP scores plot saved as: {plot_path}")
        if show_fig:
            fig.show()
    return top_vip


import numpy as np
import matplotlib.pyplot as plt

def calculate_vip(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Assumes a single response variable.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model (from sklearn.cross_decomposition)
    
    Returns
    -------
    vip : np.array of shape (n_predictors,)
        VIP scores for each predictor.
    """
def calculate_vip(pls_model):
    t = pls_model.x_scores_          # shape: (n_samples, n_components)
    q = pls_model.y_loadings_          # shape: (n_components, n_responses)
    A = t.shape[1]                    # Actual number of components
    ssy = np.zeros(A)
    
    for a in range(A):
        ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
    
    total_ssy = np.sum(ssy)
    
    vip = np.zeros(p)
    for j in range(p):
        sum_term = 0.0
        for a in range(A):
            norm_w_a = np.linalg.norm(w[:, a])
            sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
        vip[j] = np.sqrt(p * sum_term / total_ssy)
    return vip

def plot_pls_loadings(pls_model, chemical_shift, comp=1, vip_threshold=1.0, 
                      output_filename=None, show_fig=True, figure_size=(10, 6)):
    """
    Plot the loadings for a chosen component from a fitted PLS-DA model against the chemical shift axis.
    
    The loadings are plotted as a continuous trace (line) and overlaid with markers that are color-coded 
    based on their VIP scores:
      - Markers in red for VIP >= vip_threshold and positive loading.
      - Markers in blue for VIP >= vip_threshold and negative loading.
      - Markers in gray for VIP < vip_threshold.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model
        The PLS-DA model from which to extract loadings.
    chemical_shift : array-like, shape (n_predictors,)
        The chemical shift values corresponding to the predictors (should be in the same order as the predictors).
    comp : int, optional (default=1)
        The component number to plot (1 corresponds to the first component).
    vip_threshold : float, optional (default=1.0)
        The VIP threshold. Predictors with VIP >= vip_threshold will be highlighted.
    output_filename : str or None, optional (default=None)
        If provided, the plot is saved to this file.
    show_fig : bool, optional (default=True)
        If True, display the plot interactively.
    figure_size : tuple, optional (default=(10,6))
        Figure size.
    
    Returns
    -------
    None
    """
    # Convert 1-indexed comp to 0-indexed for array access.
    comp_index = comp - 1
    
    # Get the loadings for the selected component
    loadings = pls_model.x_loadings_[:, comp_index]  # shape (n_predictors,)
    
    # Calculate VIP scores for all predictors
    vip = calculate_vip(pls_model)
    
    plt.figure(figsize=figure_size)
    
    # Plot the continuous trace (line) for loadings
    plt.plot(chemical_shift, loadings, '-k', linewidth=1.5, label='Loadings Trace')
    
    # Overlay markers at each chemical shift
    for j in range(len(loadings)):
        # Choose marker color based on VIP and sign of loading
        if vip[j] >= vip_threshold:
            color = 'red' if loadings[j] >= 0 else 'blue'
            marker_size = 8
        else:
            color = 'gray'
            marker_size = 4
        plt.plot(chemical_shift[j], loadings[j], marker='s', color=color, markersize=marker_size, markeredgecolor='k')
    
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel(f'Loadings (Component {comp})')
    plt.title(f'PLS-DA Loadings (Component {comp})\nVIP Threshold = {vip_threshold}')
    
    # Create a legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=f'VIP ≥ {vip_threshold} (positive)', 
               markerfacecolor='red', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label=f'VIP ≥ {vip_threshold} (negative)', 
               markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label=f'VIP < {vip_threshold}', 
               markerfacecolor='gray', markersize=4, markeredgecolor='k'),
        Line2D([0], [0], color='k', lw=1.5, label='Loadings Trace')
    ]
    plt.legend(handles=legend_elements)
    
    # Typically, chemical shift axes are plotted in reverse (high ppm to low ppm)
    plt.gca().invert_xaxis()
    
    if output_filename is not None:
        output_path = os.path.join("images", output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved as: {output_path}")
    if show_fig:
        plt.show()
    else:
        plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_pls_loadings2(pls_model, chemical_shift, comp=1, vip_threshold=1.0, 
                      output_filename=None, show_fig=True, figure_size=(10, 6)):
    """
    Plot the loadings for a chosen component from a fitted PLS-DA model against the chemical shift axis.
    
    The loadings are plotted as a continuous trace (line) and overlaid with markers that are color-coded 
    based on their VIP scores:
      - Markers in red for VIP >= vip_threshold and positive loading.
      - Markers in blue for VIP >= vip_threshold and negative loading.
      - Markers in gray for VIP < vip_threshold.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model
        The PLS-DA model from which to extract loadings.
    chemical_shift : array-like, shape (n_predictors,)
        The chemical shift values corresponding to the predictors (should be in the same order as the predictors).
    comp : int, optional (default=1)
        The component number to plot (1 corresponds to the first component). Must be >= 1.
    vip_threshold : float, optional (default=1.0)
        The VIP threshold. Predictors with VIP >= vip_threshold will be highlighted.
    output_filename : str or None, optional (default=None)
        If provided, the plot is saved to this file.
    show_fig : bool, optional (default=True)
        If True, display the plot interactively.
    figure_size : tuple, optional (default=(10,6))
        Figure size.
    
    Returns
    -------
    None
    """
    # Ensure comp is at least 1
    if comp < 1:
        print("Warning: Component number must be >= 1. Using Component 1 instead.")
        comp = 1
    # Convert to 0-indexed for array access.
    comp_index = comp - 1

    # Get the loadings for the selected component from a PLS model.
    # For a PLSRegression, the x_loadings_ attribute holds the loadings.
    loadings = pls_model.x_loadings_  # shape (n_predictors, n_components)
    if comp_index >= loadings.shape[1]:
        raise IndexError(f"Component {comp} is out of bounds. The model only has {loadings.shape[1]} components.")
    selected_loadings = loadings[:, comp_index]  # shape (n_predictors,)

    # Calculate VIP scores for all predictors.
    # Note: calculate_vip should be defined elsewhere.
    vip = calculate_vip(pls_model)
    
    plt.figure(figsize=figure_size)
    
    # Plot the continuous trace (line) for loadings.
    plt.plot(chemical_shift, selected_loadings, '-k', linewidth=1.5, label='Loadings Trace')
    
    # Overlay markers at each chemical shift.
    for j in range(len(selected_loadings)):
        # Choose marker color based on VIP and sign of loading.
        if vip[j] >= vip_threshold:
            color = 'red' if selected_loadings[j] >= 0 else 'blue'
            marker_size = 8
        else:
            color = 'gray'
            marker_size = 4
        plt.plot(chemical_shift[j], selected_loadings[j], marker='s', color=color, 
                 markersize=marker_size, markeredgecolor='k')
    
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel(f'Loadings (Component {comp})')
    plt.title(f'PLS-DA Loadings (Component {comp})\nVIP Threshold = {vip_threshold}')
    
    # Create a legend manually.
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=f'VIP ≥ {vip_threshold} (positive)', 
               markerfacecolor='red', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label=f'VIP ≥ {vip_threshold} (negative)', 
               markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label=f'VIP < {vip_threshold}', 
               markerfacecolor='gray', markersize=4, markeredgecolor='k'),
        Line2D([0], [0], color='k', lw=1.5, label='Loadings Trace')
    ]
    plt.legend(handles=legend_elements)
    
    # Typically, chemical shift axes are plotted in reverse (high ppm to low ppm)
    plt.gca().invert_xaxis()
    
    if output_filename is not None:
        output_path = os.path.join("images", output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved as: {output_path}")
    if show_fig:
        plt.show()
    else:
        plt.close()

        
        
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder

def evaluate_plsda_components(X, y, groups=None, n_splits=5, 
                              save_fig=True, show_fig=True,
                              output_dir='images', 
                              output_file='PLSDA_Q2_R2_Scores.png',
                              figure_size=(10, 5)):
    """
    Evaluate PLS-DA performance over a range of components using cross-validation.
    
    If y is non-numeric (i.e. contains class labels), it is one-hot encoded before evaluation.
    For each number of components from 1 to the maximum allowed by both the full data and the smallest
    training set among the folds, the function computes:
      - Q² score: the mean cross-validated R² score (using GroupKFold)
      - R² score: the coefficient of determination on the full dataset.
    
    Both scores are then plotted on a single overlayed plot, with options to save and/or display the plot.
    
    Parameters:
        X (pd.DataFrame or np.array): Feature matrix.
            (If samples are in columns and features in rows, be sure to transpose beforehand if needed.)
        y (array-like or pd.Series): Target variable containing group/class labels.
            If non-numeric, y will be one-hot encoded.
        groups (array-like or None): Group labels for samples for GroupKFold.
            If None and X has an index, X.index is used.
        n_splits (int): Number of splits for GroupKFold cross-validation (default: 5).
        save_fig (bool): If True, save the combined plot as an image file.
        show_fig (bool): If True, display the plot interactively.
        output_dir (str): Directory where the plot image will be saved.
        output_file (str): Filename for the combined Q²/R² scores plot.
        figure_size (tuple): Figure size for the plot.
    
    Returns:
        q2_scores (np.array): Array of Q² scores for each number of components.
        r2_scores (np.array): Array of R² scores for each number of components.
    """
    # Use X.index as groups if groups not provided and available
    if groups is None:
        try:
            groups = X.index
        except AttributeError:
            raise ValueError("Groups not provided and X has no index attribute. Please supply groups explicitly.")
    
    # If y is a pandas Series, reindex it to match X's row order
    if isinstance(y, pd.Series):
        y = y.reindex(X.index)
    
    # If y is non-numeric, one-hot encode it
    y = np.array(y)
    if not np.issubdtype(y.dtype, np.number):
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Create GroupKFold cross-validator
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Determine maximum number of components:
    # Overall limit: n_samples - 1 or n_features
    max_comp_full = min(X.shape[0] - 1, X.shape[1])
    # Determine the smallest training set size over folds
    fold_train_sizes = []
    for train_idx, _ in group_kfold.split(X, y, groups=groups):
        fold_train_sizes.append(len(train_idx))
    min_train_size = min(fold_train_sizes)
    max_comp_cv = min(min_train_size - 1, X.shape[1])
    
    num_components = min(max_comp_full, max_comp_cv)
    if num_components < max_comp_full:
        print(f"Limiting components to {num_components} based on the smallest training fold (size={min_train_size}).")
    
    # Initialize arrays to store scores
    q2_scores = np.zeros(num_components)
    r2_scores = np.zeros(num_components)
    
    # Loop over component numbers (from 1 to num_components)
    for i in range(1, num_components + 1):
        plsr = PLSRegression(n_components=i)
        # Q²: Cross-validated R² score
        q2 = cross_val_score(plsr, X, y, cv=group_kfold, groups=groups, scoring='r2')
        q2_scores[i - 1] = np.mean(q2)
        
        # R²: Fit on full data and predict
        plsr.fit(X, y)
        y_pred = plsr.predict(X)
        r2_scores[i - 1] = r2_score(y, y_pred)
    
    # Create combined plot for Q² and R² scores
    plt.figure(figsize=figure_size)
    comps = range(1, num_components + 1)
    plt.plot(comps, q2_scores, marker='o', label='Q² Scores')
    plt.plot(comps, r2_scores, marker='o', label='R² Scores', color='orange')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('PLS-DA Q² and R² Scores')
    plt.legend()
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined Q²/R² Scores plot saved as: {output_path}")
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return q2_scores, r2_scores


def calculate_vip(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Assumes a single response variable.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model (from sklearn.cross_decomposition)
    
    Returns
    -------
    vip : np.array of shape (n_predictors,)
        VIP scores for each predictor.
    """
    t = pls_model.x_scores_  # latent scores, shape (n_samples, n_components)
    w = pls_model.x_weights_ # weights, shape (n_predictors, n_components)
    q = pls_model.y_loadings_  # y loadings, shape (n_components, 1) for single response
    A = w.shape[1]
    p = w.shape[0]
    
    # Calculate the explained sum of squares for each component.
    ssy = np.zeros(A)
    for a in range(A):
        # For a single response, q[a,0] is used.
        ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
    total_ssy = np.sum(ssy)
    
    vip = np.zeros(p)
    for j in range(p):
        sum_term = 0.0
        for a in range(A):
            norm_w_a = np.linalg.norm(w[:, a])
            sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
        vip[j] = np.sqrt(p * sum_term / total_ssy)
    return vip

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_pls_loadings(pls_model, chemical_shift, comp=0, vip_threshold=1.0, 
                      output_filename=None, show_fig=True, figure_size=(10, 6)):
    """
    Plot the loadings for a chosen component from a fitted PLS-DA model against the chemical shift axis.
    
    Variables with VIP scores greater than or equal to vip_threshold are highlighted:
      - In red if the loading is positive.
      - In blue if the loading is negative.
    All other variables are shown in gray.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model
        The PLS-DA model from which to extract loadings.
    chemical_shift : array-like, shape (n_predictors,)
        The chemical shift values corresponding to the predictors (should be in the same order as the predictors).
    comp : int, optional (default=0)
        The component index to plot (0 corresponds to the first component).
    vip_threshold : float, optional (default=1.0)
        The VIP threshold. Predictors with VIP >= vip_threshold will be highlighted.
    output_filename : str or None, optional (default=None)
        If provided, the plot is saved to this file inside the folder "images".
    show_fig : bool, optional (default=True)
        If True, display the plot interactively.
    figure_size : tuple, optional (default=(10,6))
        Figure size.
    
    Returns
    -------
    None
    """
    # Get the loadings for the selected component
    loadings = pls_model.x_loadings_[:, comp]  # shape (n_predictors,)
    
    # Calculate VIP scores for all predictors (assuming you have a calculate_vip function)
    vip = calculate_vip(pls_model)
    
    # Determine colors: if VIP >= threshold, use red for positive loadings, blue for negative; otherwise gray.
    colors = []
    for j in range(len(loadings)):
        if vip[j] >= vip_threshold:
            colors.append('red' if loadings[j] >= 0 else 'blue')
        else:
            colors.append('gray')
    
    plt.figure(figsize=figure_size)
    plt.scatter(chemical_shift, loadings, c=colors, edgecolor='k')
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel(f'Loadings (Component {comp+1})')
    plt.title(f'PLS-DA Loadings (Component {comp+1})\nVIP Threshold = {vip_threshold}')
    
    # Create a legend manually.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'VIP ≥ {vip_threshold} (positive)', 
               markerfacecolor='red', markersize=6, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label=f'VIP ≥ {vip_threshold} (negative)', 
               markerfacecolor='blue', markersize=6, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label=f'VIP < {vip_threshold}', 
               markerfacecolor='gray', markersize=4, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements)
    
    # Typically, chemical shift axes are plotted in reverse (high ppm to low ppm)
    plt.gca().invert_xaxis()
    
    # Save the figure inside the "images" folder if output_filename is provided.
    if output_filename is not None:
        folder = "images"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, output_filename)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved as: {file_path}")
        
    if show_fig:
        plt.show()
    else:
        plt.close()


import os
import plotly.graph_objects as go

def plot_pca_loadings_with_spectra(data, normalized_df, pca_model, PC_choose=1, 
                                   x_axis_col='Chemical Shift (ppm)', gap=100,
                                   spectra_scale=0.1,
                                   output_dir='images', output_file=None, 
                                   save_fig=True, show_fig=True):
    """
    Creates a Plotly figure that overlays the PCA loading (for a specified principal component)
    with the original normalized spectra in the background. The spectra are vertically offset 
    by a specified gap (default 100 units) and scaled by spectra_scale so that their intensity 
    better matches the PCA loading plot.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the x-axis values (e.g. chemical shift).
                             Used for the PCA loading trace.
        normalized_df (pd.DataFrame): DataFrame with the original normalized spectra. It must include
                                      the same x-axis column (x_axis_col) and one or more spectra columns.
        pca_model: Fitted PCA model from which the loadings will be extracted.
        PC_choose (int): Principal component number for which to plot the loading.
        x_axis_col (str): Name of the x-axis column (default 'Chemical Shift (ppm)').
        gap (float): Vertical gap (offset) between successive spectra.
        spectra_scale (float): Factor to multiply the normalized spectra intensities by.
        output_dir (str): Directory to save the plot HTML file.
        output_file (str or None): Filename for the output HTML file. If None, a default name is generated.
        save_fig (bool): If True, save the figure as an HTML file.
        show_fig (bool): If True, display the interactive figure.
    
    Returns:
        fig: The Plotly figure object.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings_with_Spectra.html'
        
    fig = go.Figure()

    # Add the original normalized spectra traces (background)
    # We assume normalized_df has the x-axis column, and all other columns are spectra.
    spectra_cols = [col for col in normalized_df.columns if col != x_axis_col]
    for i, col in enumerate(spectra_cols):
        offset = i * gap
        fig.add_trace(go.Scatter(
            x=normalized_df[x_axis_col],
            y=(normalized_df[col] * spectra_scale) + offset,
            mode='lines',
            line=dict(color='gray'),
            name=f'{col} (offset {offset})',
            opacity=0.6,
            showlegend=False  # Hide individual legend entries for clarity
        ))

    # Add the PCA loading trace (overlayed on top)
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'PC{PC_choose} Loading'
    ))
    
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot with Original Spectra Overlay',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value / Offset Spectra',
        legend_title='Component',
        xaxis=dict(autorange='reversed')
    )
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig
  

# Hierarchical Cluster Analysis (HCA)
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, fcluster

def perform_hca(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="NMR_filename",
    method="ward",
    metric="euclidean",
    n_clusters=None,
    output_dir="images",
    dendrogram_filename=None,
    show_fig=True
):
    """
    Perform Hierarchical Cluster Analysis (HCA) on the sample data and generate an interactive dendrogram.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples and rows as features.
    metadata : pd.DataFrame
        Metadata DataFrame containing sample information. Must include the sample_id_col and group_col.
    group_col : str
        Column in metadata that contains group labels.
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns.
    method : str
        The linkage algorithm to use (e.g., "ward", "single", "complete"). Default is "ward".
    metric : str
        The distance metric to use (e.g., "euclidean", "cosine"). Default is "euclidean".
    n_clusters : int or None
        If provided, the function will assign cluster memberships using the specified number of clusters.
    output_dir : str
        Directory where the dendrogram HTML file will be saved if dendrogram_filename is provided.
    dendrogram_filename : str or None
        If provided, the interactive dendrogram is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly dendrogram is displayed.
    
    Returns
    -------
    result_dict : dict
        Dictionary containing:
          - "dendrogram_fig": The Plotly figure for the dendrogram.
          - "linkage_matrix": The linkage matrix computed for clustering.
          - "cluster_assignments": (Optional) A pandas Series mapping sample IDs to their cluster labels (if n_clusters is provided).
    """
    import os
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, fcluster
    import plotly.figure_factory as ff

    # Transpose data so that rows represent samples
    X = data.drop(columns="Chemical Shift (ppm)").transpose()  # shape: [n_samples, n_features]
    
    # Compute the linkage matrix using the specified method and metric
    Z = linkage(X, method=method, metric=metric)
    
    # Create a dendrogram figure using Plotly's figure factory
    # The labels are the sample IDs from X.index.
    dendro_fig = ff.create_dendrogram(
        X.values,
        orientation='left',
        labels=list(X.index),
        linkagefun=lambda x: Z,
        color_threshold=0  # Let Plotly choose the default color cutoff
    )
    
    # Merge metadata for hover labels
    meta_lookup = metadata.set_index(sample_id_col)[group_col].to_dict()
    for trace in dendro_fig.data:
        if trace.text is not None:
            trace.text = [f"{lbl}<br>Group: {meta_lookup.get(lbl, 'NA')}" for lbl in trace.text]
    
    # If n_clusters is specified, assign clusters using fcluster.
    cluster_assignments = None
    if n_clusters is not None:
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        # Map sample IDs to clusters
        cluster_assignments = pd.Series(clusters, index=X.index, name='Cluster')
        print("Cluster assignments:")
        print(cluster_assignments)
    
    # Save the dendrogram if a filename is provided.
    if dendrogram_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, dendrogram_filename)
        dendro_fig.write_html(output_path)
        print(f"Dendrogram saved as: {output_path}")
    
    if show_fig:
        dendro_fig.show()
    
    result_dict = {
        "dendrogram_fig": dendro_fig,
        "linkage_matrix": Z,
        "cluster_assignments": cluster_assignments
    }
    return result_dict

import os
import csv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def perform_opls_da(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="NMR_filename",
    n_components=2,
    output_dir="images",
    score_plot_filename=None,
    show_fig=True
):
    """
    Perform Orthogonal PLS-DA (OPLS-DA) by:
      1. One-hot encoding the class labels.
      2. Fitting a PLSRegression model with one predictive component.
      3. Removing the predictive variation to get orthogonal components using PCA on the residual matrix.
      4. Plotting the predictive (LV1) vs. the first orthogonal component (LV2).

    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples (rows as features).
        (If the chemical shift axis is included as a column or row, it will be ignored.)
    metadata : pd.DataFrame
        DataFrame containing sample metadata. Must include the sample_id_col (matching data columns)
        and a grouping column (group_col) for classification.
    group_col : str
        Column in metadata that contains the class/group labels.
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns.
    n_components : int
        Total number of components: 1 predictive + (n_components - 1) orthogonal.
        Default is 2 so that a 2D score plot can be generated.
    output_dir : str
        Directory where the HTML plot will be saved if score_plot_filename is provided.
    score_plot_filename : str or None
        If provided, the interactive score plot is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly figure is displayed.

    Returns
    -------
    model_dict : dict
        Dictionary containing:
          - "pls_model": the fitted PLSRegression model (predictive component).
          - "pca_ortho": the PCA model fitted on the residual (orthogonal components) (or None if n_components==1).
          - "scores_df": DataFrame of latent variable scores (predictive and orthogonal) merged with metadata.
    """
    # If data is a DataFrame, remove any column or row named "Chemical Shift (ppm)"
    if isinstance(data, pd.DataFrame):
        if "Chemical Shift (ppm)" in data.columns:
            data = data.drop(columns=["Chemical Shift (ppm)"])
        if "Chemical Shift (ppm)" in data.index:
            data = data.drop(index=["Chemical Shift (ppm)"])
    
    # Ensure output directory exists if saving the plot.
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Transpose data so that rows are samples and columns are features.
    X = data.transpose()  # shape: [n_samples, n_features]

    # Align metadata with the sample IDs from X.
    sample_index_df = pd.DataFrame({sample_id_col: X.index})
    merged_df = sample_index_df.merge(metadata, on=sample_id_col, how='left')

    # One-hot encode the group labels.
    groups = merged_df[group_col].astype(str).values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(groups)  # shape: [n_samples, n_classes]

    # --- Step 1: Fit PLSRegression for the predictive component ---
    pls_model = PLSRegression(n_components=1)
    pls_model.fit(X, Y)
    t_pred = pls_model.x_scores_      # predictive scores; shape: [n_samples, 1]
    p_pred = pls_model.x_loadings_    # predictive loadings; shape: [n_features, 1]

    # --- Step 2: Remove predictive variation to compute orthogonal variation ---
    # Reconstruct X from the predictive component.
    X_pred = t_pred.dot(p_pred.T)
    # Residual matrix (X_res) contains the orthogonal variation.
    X_res = X - X_pred

    # --- Step 3: Extract orthogonal component(s) from the residual using PCA ---
    n_ortho = n_components - 1  # number of orthogonal components
    if n_ortho > 0:
        pca_ortho = PCA(n_components=n_ortho)
        t_ortho = pca_ortho.fit_transform(X_res)  # shape: [n_samples, n_ortho]
    else:
        pca_ortho = None
        t_ortho = np.empty((X.shape[0], 0))
    
    # --- Step 4: Build the scores DataFrame ---
    # Concatenate the predictive score with orthogonal scores.
    scores = np.concatenate([t_pred, t_ortho], axis=1)  # shape: [n_samples, n_components]
    lv_names = ['Predictive'] + [f"Orthogonal{i+1}" for i in range(n_ortho)]
    scores_df = pd.DataFrame(scores, columns=lv_names, index=X.index)
    scores_df = scores_df.reset_index().rename(columns={"index": sample_id_col})
    scores_df = scores_df.merge(
        merged_df[[sample_id_col, group_col]],
        on=sample_id_col,
        how='left'
    )
    
    # --- Step 5: Plot the score plot (Predictive vs. first Orthogonal) ---
    if n_components >= 2:
        fig = px.scatter(
            scores_df,
            x="Predictive",
            y="Orthogonal1",
            color=group_col,
            hover_data=[sample_id_col],
            title="OPLS-DA Score Plot (Predictive vs. Orthogonal)",
            labels={"Predictive": "Predictive Component", "Orthogonal1": "Orthogonal Component 1"}
        )
        if score_plot_filename is not None:
            html_path = os.path.join(output_dir, score_plot_filename)
            fig.write_html(html_path)
        if show_fig:
            fig.show()
    else:
        print("n_components < 2, so no 2D score plot was generated.")
    
    model_dict = {"pls_model": pls_model, "pca_ortho": pca_ortho, "scores_df": scores_df}
    return model_dict


def plot_oplsda_predictive_loadings(data, model_dict, x_axis_col='Chemical Shift (ppm)', 
                                    output_dir='images', output_file=None, 
                                    save_fig=True, show_fig=True):
    """
    Plot the predictive loadings from the OPLS-DA model.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the x-axis values (e.g., retention time, chemical shift).
    model_dict : dict
        Dictionary returned by perform_opls_da.
    x_axis_col : str
        Column in data to use for the x-axis.
    output_dir : str
        Directory where the plot will be saved.
    output_file : str or None
        Filename for saving the plot. Defaults to 'OPLSDA_Predictive_Loadings.html' if None.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The predictive loadings plot.
    """
    # Predictive loadings from the PLS model.
    loadings = model_dict["pls_model"].x_loadings_
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings[:, 0],
        mode='lines',
        name='OPLS-DA Predictive Loading'
    ))
    fig.update_layout(
        title='OPLS-DA Predictive Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        xaxis=dict(autorange='reversed')
    )
    if save_fig:
        if output_file is None:
            output_file = 'OPLSDA_Predictive_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Predictive loadings plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def plot_oplsda_orthogonal_loadings(data, model_dict, component=1, x_axis_col='RT(min)', 
                                    output_dir='images', output_file=None, 
                                    save_fig=True, show_fig=True):
    """
    Plot the orthogonal loadings from the OPLS-DA model (from the PCA on the residual).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the x-axis values (e.g., retention time, chemical shift).
    model_dict : dict
        Dictionary returned by perform_opls_da.
    component : int
        The orthogonal component number to plot (1-indexed).
    x_axis_col : str
        Column in data to use for the x-axis.
    output_dir : str
        Directory where the plot will be saved.
    output_file : str or None
        Filename for saving the plot. Defaults to 'OPLSDA_Orthogonal_Component_1_Loadings.html' if None.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The orthogonal loadings plot.
    """
    pca_ortho = model_dict.get("pca_ortho", None)
    if pca_ortho is None:
        raise ValueError("No orthogonal components are available in the OPLS-DA model.")
    
    if component < 1 or component > pca_ortho.components_.shape[0]:
        raise ValueError(f"Component must be between 1 and {pca_ortho.components_.shape[0]}, got {component}")
    
    # PCA components are stored in rows; select the desired orthogonal component.
    loadings = pca_ortho.components_[component - 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings,
        mode='lines',
        name=f'OPLS-DA Orthogonal Component {component} Loading'
    ))
    fig.update_layout(
        title=f'OPLS-DA Orthogonal Component {component} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        xaxis=dict(autorange='reversed')
    )
    if save_fig:
        if output_file is None:
            output_file = f'OPLSDA_Orthogonal_Component_{component}_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Orthogonal loadings plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def calculate_opls_vip_scores(model_dict, X):
    """
    Calculate VIP (Variable Importance in Projection) scores for the predictive component
    from the OPLS-DA model.
    
    Parameters
    ----------
    model_dict : dict
        Dictionary returned by perform_opls_da.
    X : pd.DataFrame
        DataFrame of features (with samples as rows). The columns should correspond to the variables.
    
    Returns
    -------
    vip : np.array
        Array of VIP scores for each variable.
    """
    pls_model = model_dict["pls_model"]
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip

def analyze_opls_vip_scores(model_dict, X, top_n=10,
                            save_df=False, output_dir='images', output_file_df=None,
                            plot_fig=False, save_fig=False, output_file_plot=None, show_fig=True):
    """
    Analyze and optionally save/plot the top VIP scores for the predictive component from the OPLS-DA model.
    
    Parameters
    ----------
    model_dict : dict
        Dictionary returned by perform_opls_da.
    X : pd.DataFrame
        Feature DataFrame with variables as columns.
    top_n : int
        Number of top VIP scores to return.
    save_df : bool
        Whether to save the VIP scores as a CSV file.
    output_dir : str
        Directory for saving outputs.
    output_file_df : str or None
        Filename for saving the VIP scores CSV.
    plot_fig : bool
        Whether to plot the VIP scores.
    save_fig : bool
        Whether to save the VIP scores plot.
    output_file_plot : str or None
        Filename for the VIP scores plot.
    show_fig : bool
        Whether to display the plot.
    
    Returns
    -------
    top_vip : pd.DataFrame
        DataFrame of the top VIP scores.
    """
    vip = calculate_opls_vip_scores(model_dict, X)
    vip_df = pd.DataFrame({'Variable': X.columns, 'VIP Score': vip})
    vip_df = vip_df.sort_values(by='VIP Score', ascending=False)
    top_vip = vip_df.head(top_n)
    if save_df:
        if output_file_df is None:
            output_file_df = 'OPLS_VIP_scores.csv'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file_df)
        top_vip.to_csv(output_path, index=False)
        print(f"VIP scores saved as: {output_path}")
    if plot_fig:
        fig = px.bar(top_vip, x='Variable', y='VIP Score',
                     title='Top VIP Scores (OPLS-DA)',
                     text='VIP Score')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        if save_fig:
            if output_file_plot is None:
                output_file_plot = 'OPLS_VIP_scores_plot.html'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, output_file_plot)
            fig.write_html(plot_path)
            print(f"VIP scores plot saved as: {plot_path}")
        if show_fig:
            fig.show()
    return top_vip





def plot_hca_heatmap(
    data,
    metadata,
    sample_id_col="NMR_filename",
    group_col="ATTRIBUTE_group",
    output_dir="images",
    heatmap_filename=None,
    show_fig=True
):
    """
    Generate a heatmap of the feature matrix with samples clustered by hierarchical clustering.
    Samples are ordered based on the dendrogram.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples.
    metadata : pd.DataFrame
        Metadata DataFrame containing sample information.
    sample_id_col : str
        Column in metadata that contains sample IDs.
    group_col : str
        Column in metadata that contains group labels.
    output_dir : str
        Directory where the heatmap HTML file will be saved if heatmap_filename is provided.
    heatmap_filename : str or None
        If provided, the heatmap is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive heatmap is displayed.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The heatmap figure.
    """
    import plotly.express as px
    # Transpose data so that rows are samples
    X = data.transpose()  # shape: [n_samples, n_features]
    
    # Compute hierarchical clustering order
    Z = linkage(X, method="ward", metric="euclidean")
    # Use the dendrogram to get the order of sample indices
    dendro = ff.create_dendrogram(X.values, orientation='left', linkagefun=lambda x: Z)
    ordered_sample_ids = dendro['layout']['yaxis']['ticktext']
    
    # Reorder the DataFrame based on the dendrogram order
    X_ordered = X.loc[ordered_sample_ids]
    
    # Add group information from metadata for annotation
    meta_lookup = metadata.set_index(sample_id_col)[group_col].to_dict()
    group_labels = [meta_lookup.get(sid, 'NA') for sid in X_ordered.index]
    
    # Create a heatmap using Plotly Express
    fig = px.imshow(
        X_ordered,
        labels=dict(x="Features", y="Samples", color="Intensity"),
        x=X_ordered.columns,
        y=X_ordered.index,
        aspect="auto",
        title="HCA Heatmap"
    )
    # Add group labels as hover information
    fig.update_traces(
        hovertemplate="<b>Sample</b>: %{y}<br><b>Group</b>: " +
                      "%{customdata}<br><b>Feature</b>: %{x}<br><b>Intensity</b>: %{z}"
    )
    fig.update_traces(customdata=np.array(group_labels).reshape(-1, 1))
    
    if heatmap_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, heatmap_filename)
        fig.write_html(output_path)
        print(f"Heatmap saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig

# --------------------------------------------------------------------------
#               9) STOCSY FUNCTIONS
# --------------------------------------------------------------------------
def STOCSY(target, X, ppm):
    import mpld3
    from matplotlib.collections import LineCollection  # import directly
    if type(target) == float:
        idx = np.abs(ppm - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target
    corr = (stats.zscore(target_vect.T, ddof=1) @ stats.zscore(X.T, ddof=1)) / ((X.T.shape[0]) - 1)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / ((X.T.shape[0]) - 1)
    
    x = np.linspace(0, len(covar), len(covar))
    y = covar
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(16,4))
    norm = plt.Normalize(corr.min(), corr.max())
    # Use LineCollection from matplotlib.collections (aliased as LineCollection)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min(), y.max())
    axs.invert_xaxis()
    
    minppm = min(ppm)
    maxppm = max(ppm)
    ticksx = []
    tickslabels = []
    if maxppm < 30:
        ticks = np.linspace(int(math.ceil(minppm)), int(maxppm), int(maxppm) - math.ceil(minppm) + 1)
    else:
        ticks = np.linspace(int(math.ceil(minppm / 10.0)) * 10, (int(math.ceil(maxppm / 10.0)) * 10) - 10, int(math.ceil(maxppm / 10.0)) - int(math.ceil(minppm / 10.0)))
    currenttick = 0
    for ppm_val in ppm:
        if currenttick < len(ticks) and ppm_val > ticks[currenttick]:
            position = int((ppm_val - minppm) / (maxppm - minppm) * max(x))
            if position < len(x):
                ticksx.append(x[position])
                tickslabels.append(ticks[currenttick])
            currenttick += 1
    plt.xticks(ticksx, tickslabels, fontsize=12)
    
    axs.set_xlabel('ppm', fontsize=14)
    axs.set_ylabel(f"Covariance with \n signal at {target:.2f} ppm", fontsize=14)
    axs.set_title(f'STOCSY from signal at {target:.2f} ppm', fontsize=16)
    
    text = axs.text(1, 1, '')
    lnx = plt.plot([60,60], [0,1.5], color='black', linewidth=0.3)
    lny = plt.plot([0,100], [1.5,1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')
    
    def hover(event):
        if event.inaxes == axs:
            inv = axs.transData.inverted()
            maxcoord = axs.transData.transform((x[0], 0))[0]
            mincoord = axs.transData.transform((x[-1], 0))[0]
            ppm_val = ((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * (maxppm - minppm) + minppm
            cov_val = covar[int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(covar))]
            cor_val = corr[int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(corr))]
            text.set_visible(True)
            text.set_position((event.xdata, event.ydata))
            text.set_text('{:.2f}'.format(ppm_val) + " ppm, covariance: " + '{:.6f}'.format(cov_val) + ", correlation: " + '{:.2f}'.format(cor_val))
            lnx[0].set_data([event.xdata, event.xdata], [-1, 1])
            lnx[0].set_linestyle('--')
            lny[0].set_data([x[0], x[-1]], [cov_val, cov_val])
            lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)    
    plt.show()
    
    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}.pdf", transparent=True, dpi=300)
    
    html_str = mpld3.fig_to_html(fig)
    with open(f"images/stocsy_interactive_{target}ppm.html", "w") as f:
        f.write(html_str)
    
    return corr, covar


def STOCSY_interactive(target, X, ppm):
    if isinstance(target, float):
        idx = np.argmin(np.abs(np.array(ppm) - target))
        target_vect = X.iloc[idx]
    else:
        target_vect = target
    corr = (stats.zscore(target_vect.T, ddof=1) @ stats.zscore(X.T, ddof=1)) / ((X.T.shape[0]) - 1)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / ((X.T.shape[0]) - 1)
    df = pd.DataFrame({'ppm': ppm, 'covar': covar, 'corr': corr})
    df = df.sort_values('ppm', ascending=False)
    fig = go.Figure(data=go.Scatter(
        x=df['ppm'],
        y=df['covar'],
        mode='lines+markers',
        marker=dict(
            color=df['corr'],
            colorscale='jet',
            colorbar=dict(title='Correlation'),
            size=6
        ),
        line=dict(width=2),
        hovertemplate='Chemical Shift: %{x:.2f} ppm<br>Covariance: %{y:.6f}<br>Correlation: %{marker.color:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'STOCSY from signal at {target:.2f} ppm',
        xaxis_title='ppm',
        yaxis_title=f"Covariance with signal at {target:.2f} ppm",
        xaxis_autorange='reversed'
    )
    fig.show()
    return corr, covar

def STOCSY_mode(target, X, rt_values, mode="linear"):
    """
    Structured STOCSY: Compute correlation (or similarity), R², and RMSE between a target signal and a matrix of signals.
    
    Parameters:
    ----------
    target : float or Series
        Target Chemical Shift (ppm) (if float) or signal vector for STOCSY anchor.
    X : DataFrame
        Data matrix where each row is a signal (e.g., from NMR).
    rt_values : Series
        Chemical Shift (ppm) values corresponding to each row in X.
    mode : str, optional
        Type of structured correlation model to use:
            - 'linear'       : Pearson correlation (default).
            - 'exponential'  : Exponential decay (kinetics, degradation).
            - 'sinusoidal'   : Sine wave (circadian rhythms, periodicity; uses FFT to estimate frequency).
            - 'sigmoid'      : Logistic dose-response (biological saturation, enzyme kinetics; captures thresholds and plateaus).
            - 'gaussian'     : Symmetric bell curve (chromatographic peaks, transient events).
            - 'fft'          : Frequency domain similarity (hidden periodic structure, using cosine similarity of FFT magnitudes).
            - 'polynomial'   : 2nd-degree polynomial trend (curvilinear responses).
            - 'piecewise'    : Piecewise linear (threshold, on/off behavior).
            - 'skewed_gauss' : Asymmetric bell-shaped peak (real-world chromatographic peaks).
    
    Returns:
    -------
    corr : array
        Array of correlation (or similarity) values between the target and each signal.
    covar : array
        Array of covariance values between the target and each signal.
    """
    import os
    import mpld3
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy import stats
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import cosine
    from scipy.fft import fft
    from scipy.stats import skewnorm

    # Define models
    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    def sin_model(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def sigmoid_model(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def gauss_model(x, a, mu, sigma, c):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    def poly_model(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x**2

    def piecewise_model(x, x0, k1, b1, k2, b2):
        return np.piecewise(x, [x < x0, x >= x0],
                            [lambda x: k1 * x + b1, lambda x: k2 * x + b2])
    
    def skewed_gauss_model(x, a, loc, scale, alpha, offset):
        # skewnorm.pdf returns density; multiplicative factor a and offset applied.
        return a * skewnorm.pdf(x, alpha, loc, scale) + offset

    def fft_magnitude(signal):
        return np.abs(fft(signal))

    # Determine target vector
    if isinstance(target, float):
        idx = np.abs(rt_values - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target

    corr_list = []
    r2_list = []
    rmse_list = []
    target_vals = target_vect.values

    # Loop por cada sinal em X
    for i in range(X.shape[0]):
        y = X.iloc[i].values
        x = target_vals  # usamos o mesmo vetor de x do sinal target

        try:
            if mode == "linear":
                r = np.corrcoef(x, y)[0, 1]
                fitted = y

            elif mode == "exponential":
                popt, _ = curve_fit(exp_model, x, y, maxfev=10000)
                fitted = exp_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "sinusoidal":
                if np.std(y) < 1e-6:
                    raise ValueError("Low signal variance")
                # Remover tendência e estimar frequência dominante via FFT
                y_detrended = y - np.mean(y)
                fft_y = fft(y_detrended)
                freqs = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
                # Ignorar a frequência 0
                dom_freq_index = np.argmax(np.abs(fft_y[1:])) + 1
                dom_freq = np.abs(freqs[dom_freq_index])
                guess_freq = 2 * np.pi * dom_freq if dom_freq > 0 else 1 / (2 * np.pi)
                popt, _ = curve_fit(sin_model, x, y, p0=[1, guess_freq, 0, 0], maxfev=10000)
                fitted = sin_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "sigmoid":
                if np.max(x) - np.min(x) == 0 or np.max(y) - np.min(y) == 0:
                    raise ValueError("Flat signal")
                x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
                y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
                popt, _ = curve_fit(sigmoid_model, x_scaled, y_scaled, p0=[1, 1, 0.5], maxfev=10000)
                fitted_scaled = sigmoid_model(x_scaled, *popt)
                fitted = fitted_scaled * (np.max(y) - np.min(y)) + np.min(y)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "gaussian":
                mu_init = x[np.argmax(y)]
                half_max = np.max(y) / 2
                indices = np.where(y > half_max)[0]
                if len(indices) > 1:
                    sigma_init = (x[indices[-1]] - x[indices[0]]) / 2.355  # FWHM ~2.355σ
                else:
                    sigma_init = np.std(x)
                popt, _ = curve_fit(gauss_model, x, y, p0=[1, mu_init, sigma_init, 0], maxfev=10000)
                fitted = gauss_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "fft":
                # Compute FFT magnitudes and compare via cosine similarity
                target_fft = fft_magnitude(target_vals)
                y_fft = fft_magnitude(y)
                similarity = 1 - cosine(target_fft, y_fft)
                r = similarity
                fitted = np.zeros_like(y)  # Não há ajuste, apenas similaridade

            elif mode == "polynomial":
                popt, _ = curve_fit(poly_model, x, y, p0=[0, 1, 0], maxfev=10000)
                fitted = poly_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "piecewise":
                # Estimativa inicial: ponto de mudança no meio do domínio
                x0_guess = np.median(x)
                popt, _ = curve_fit(piecewise_model, x, y, p0=[x0_guess, 1, 0, 0.5, 0], maxfev=10000)
                fitted = piecewise_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "skewed_gauss":
                loc_init = x[np.argmax(y)]
                scale_init = np.std(x)
                popt, _ = curve_fit(skewed_gauss_model, x, y, p0=[1, loc_init, scale_init, 4, 0], maxfev=10000)
                fitted = skewed_gauss_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            else:
                raise ValueError("Invalid mode")
        except Exception:
            r = 0
            fitted = np.zeros_like(y)

        r2 = r ** 2
        rmse = np.sqrt(np.mean((y - fitted) ** 2)) if mode != "fft" else np.nan
        corr_list.append(r)
        r2_list.append(r2)
        rmse_list.append(rmse)

    corr = np.array(corr_list)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / (X.T.shape[0] - 1)

    # Plotting dos resultados (mantido igual, com eixo X invertido)
    x_plot = np.linspace(0, len(covar), len(covar))
    y_plot = covar
    points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4), sharex=True, sharey=True)
    norm = plt.Normalize(corr.min(), corr.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax)

    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_ylim(y_plot.min(), y_plot.max())
    ax.invert_xaxis()

    # Configurar os ticks do eixo X com base nos rt_values
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    ticksx = []
    tickslabels = []
    if max_rt < 30:
        ticks = np.linspace(math.ceil(min_rt), int(max_rt), int(max_rt) - math.ceil(min_rt) + 1)
    else:
        ticks = np.linspace(math.ceil(min_rt / 10.0) * 10,
                             math.ceil(max_rt / 10.0) * 10 - 10,
                             math.ceil(max_rt / 10.0) - math.ceil(min_rt / 10.0))
    currenttick = 0
    for rt_val in rt_values:
        if currenttick < len(ticks) and rt_val > ticks[currenttick]:
            position = int((rt_val - min_rt) / (max_rt - min_rt) * x_plot.max())
            if position < len(x_plot):
                ticksx.append(x_plot[position])
                tickslabels.append(ticks[currenttick])
            currenttick += 1
    plt.xticks(ticksx, tickslabels, fontsize=12)

    ax.set_xlabel('Chemical Shift (ppm)', fontsize=14)
    ax.set_ylabel(f"Covariance with \n signal at {target:.2f} ppm", fontsize=14)
    ax.set_title(f'STOCSY from signal at {target:.2f} ppm ({mode} model)', fontsize=16)

    # Configurar função hover para visualização interativa
    text = ax.text(1, 1, '')
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')

    def hover(event):
        if event.inaxes == ax:
            inv = ax.transData.inverted()
            maxcoord = ax.transData.transform((x_plot[0], 0))[0]
            mincoord = ax.transData.transform((x_plot[-1], 0))[0]
            rt_val = ((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * (max_rt - min_rt) + min_rt
            i = int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(covar))
            if 0 <= i < len(covar):
                cov_val = covar[i]
                cor_val = corr[i]
                text.set_visible(True)
                text.set_position((event.xdata, event.ydata))
                text.set_text(f'{rt_val:.2f} min, covariance: {cov_val:.6f}, correlation: {cor_val:.2f}')
                lnx[0].set_data([event.xdata, event.xdata], [-1, 1])
                lnx[0].set_linestyle('--')
                lny[0].set_data([x_plot[0], x_plot[-1]], [cov_val, cov_val])
                lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}min_{mode}.pdf", transparent=True, dpi=300)

    import mpld3
    html_str = mpld3.fig_to_html(fig)
    with open(f"images/stocsy_interactive_{target}min_{mode}.html", "w") as f:
        f.write(html_str)

    plt.show()
    return corr, covar


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplots_by_class(
    auc_df, 
    df_metadata, 
    peaks,
    classification_column='ATTRIBUTE_group',
    output_folder='images', 
    filename='boxplot_auc_by_class.png',
    show_plot=True
):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Garante que a pasta existe
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)

    # Renomeia as colunas de auc_df com base nos pares de picos
    auc_df = auc_df.copy()
    region_labels = [f"{peaks[i]:.2f}–{peaks[i+1]:.2f}" for i in range(0, len(peaks), 2)]
    auc_df.columns = region_labels
    auc_df['NMR_filename'] = auc_df.index

    # Faz o merge com o metadata
    metadata_subset = df_metadata[['NMR_filename', classification_column]].copy()
    merged_df = pd.merge(auc_df, metadata_subset, on='NMR_filename')

    if merged_df.empty:
        print("⚠️ Nenhum dado foi unido. Verifique se auc_df.index coincide com df_metadata['NMR_filename'].")
        return

    # Transforma para formato longo para seaborn
    melted_df = merged_df.melt(id_vars=[classification_column], 
                               value_vars=region_labels,
                               var_name='Region', value_name='AUC')

    # Cria o boxplot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Region', y='AUC', hue=classification_column, data=melted_df)
    plt.title('AUC por Região agrupado por Classificação')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salva e exibe
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()



# --------------------------------------------------------------------------
#                    10) Data-Export
# --------------------------------------------------------------------------

import os
import csv
import re

def sanitize_string(s):
    """
    Sanitize a string so that it contains only English letters, numbers, and underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def export_metaboanalyst(aligned_df, df_metadata,
                         sample_id_col="NMR_filename",
                         class_col="ATTRIBUTE_classification",
                         output_file="metaboanalyst_input.csv"):
    """
    Export NMR data to a CSV suitable for MetaboAnalyst with the following format:
      - First row: column headers (e.g., "Chemical Shift (ppm)" plus sample IDs)
      - Second row: blank under "Chemical Shift (ppm)", then each sample's classification
      - Remaining rows: the actual data from aligned_df

    The function filters the metadata to include only samples present in the data.
    It also sanitizes sample IDs and classification values so that only English letters,
    numbers, and underscores remain.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        NMR data with the first column "Chemical Shift (ppm)" and subsequent columns as sample intensities.
        The column names after the first one must correspond to sample IDs.
    df_metadata : pd.DataFrame
        Metadata DataFrame containing sample information. Must include:
          - sample_id_col: the column with sample IDs.
          - class_col: the column with class/group information for each sample.
    sample_id_col : str, optional
        The column in df_metadata that contains sample IDs.
    class_col : str, optional
        The column in df_metadata that contains the sample classification.
    output_file : str, optional
        The output CSV filename.

    Returns
    -------
    new_df : pd.DataFrame
        The new DataFrame (with only valid sample columns) that is saved to CSV.
    """
    # --- Step 1: Sanitize column headers in aligned_df ---
    # Assume the first column is "Chemical Shift (ppm)" and keep it unchanged.
    orig_col_names = list(aligned_df.columns)
    # For sample columns (all except the first), sanitize their names.
    sanitized_sample_cols = [sanitize_string(s) for s in orig_col_names[1:]]
    sanitized_col_names = [orig_col_names[0]] + sanitized_sample_cols
    aligned_df = aligned_df.copy()
    aligned_df.columns = sanitized_col_names

    # --- Step 2: Identify sample columns (all except the first "Chemical Shift (ppm)") ---
    sample_cols = sanitized_col_names[1:]
    
    # --- Step 3: Filter and sanitize metadata ---
    # Create a copy of the metadata and sanitize the sample IDs.
    meta = df_metadata.copy()
    meta[sample_id_col] = meta[sample_id_col].apply(sanitize_string)
    # Also sanitize the classification values.
    meta[class_col] = meta[class_col].apply(sanitize_string)
    # Set the index using the sanitized sample IDs.
    meta_indexed = meta.set_index(sample_id_col)
    # Reindex metadata to include only sample IDs from aligned_df.
    classification_series = meta_indexed.reindex(sample_cols)[class_col]
    
    # Optionally, drop any sample columns with missing metadata.
    valid_sample_cols = classification_series.dropna().index.tolist()
    if len(valid_sample_cols) < len(sample_cols):
        missing = set(sample_cols) - set(valid_sample_cols)
        print(f"Warning: The following sample IDs are missing metadata and will be excluded: {missing}")
    
    # --- Step 4: Build the classification row ---
    # First cell is blank (for "Chemical Shift (ppm)"), then classification values for valid sample columns.
    classification_row = [""]
    classification_row.extend(classification_series.loc[valid_sample_cols].values)
    
    # --- Step 5: Build a new DataFrame using only "Chemical Shift (ppm)" and valid sample columns ---
    new_df = aligned_df[['Chemical Shift (ppm)'] + valid_sample_cols].copy()
    
    # --- Step 6: Write the CSV with the classification row after the header ---
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write header row.
        writer.writerow(new_df.columns.tolist())
        # Write classification row.
        writer.writerow(classification_row)
        # Write data rows.
        for i in range(len(new_df)):
            writer.writerow(new_df.iloc[i].values)
    
    print(f"MetaboAnalyst input CSV saved as: {output_file}")
    return new_df


# --------------------------------------------------------------------------
#                    11) Data-Processing Report
# --------------------------------------------------------------------------
def print_data_processing_report(start_rt, end_rt, samples_to_remove,
                                 aligned_method, normalization_method,
                                 scale_method):
    """
    Print a concise data-processing report for LC analysis.
    """
    samples_str = ", ".join(samples_to_remove) if samples_to_remove else "None"
    report = f"""
Data-Processing Report
----------------------
1. Unwanted outer RT regions removed: {start_rt}–{end_rt} min
2. Removed samples: {samples_str}
3. Alignment method: {aligned_method}
4. Normalization method: {normalization_method}
5. Scaling method: {scale_method}
    """
    print(report.strip())

# --------------------------------------------------------------------------
#                       End of data_processing_NMR.py
# --------------------------------------------------------------------------
