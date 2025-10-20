# single_total_select.py
# ------------------------------------------------------------
# Single Mode: simulación de TRAIN (todos los años) + VALIDACIÓN.
# Selección del mejor miembro usando MAE/PSD/ACF calculados
#   exclusivamente sobre el *año anterior al periodo de validación*.
# Guarda .TXT (TSV) para TOTAL (hist+val) y para SOLO VALIDACIÓN.
# Opciones nuevas:
#   - PRE_WAVELET_FILTER_ENABLE: filtra la serie de precios al inicio (SWT).
#   - ROLLING_HISTORICAL_SIM   : simula histórico en modo rolling (Y1=real; Y2~train(Y1); Y3~train(Y1+Y2); ...)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import warnings


# ==================== I/O y guardado ====================
SAVE_LAST_SIM   = True
SIM_SAVE_DIR    = "saved_sims"
SIM_SAVE_PREFIX = "LASTSIM_4_"

# ==================== Modo Single ====================
ENSEMBLE_SIZE = 200
SINGLE_CONFIG = dict(
    rc=True,   # recentrado post-mapeo
    hl=False,   # heterocedasticidad simple por hora
    psd=True,  # moldear PSD iterativamente
    clip=True, # clip en PSD
    conv=True, # criterio de convergencia
    wv=True    # mapeo de cuantiles guiado por wavelet
)

# ==================== Pesos de selección ====================
# Se normalizan internamente; si todos son 0 => w_mae=1.
SELECT_WEIGHTS = dict(mae=70.0, psd=10.0, acf=20.0)

# ==================== Datos ====================
TRAIN_PATHS = [
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2018.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2019.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2020.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2021.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2022.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2023.tsv.TXT",
]
VAL_PATH = r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2024.tsv.TXT"

# ==================== Constantes ====================
HOLIDAYS = []
LOG_OFFSET_C = 0.5
PSD_CLIP_RANGE = (0.1, 10)
PSD_MAX_ITER   = 50
PSD_CONV_TOL   = 1e-4

# Wavelet / SWT (para qmaps y para filtro opcional)
WV_ENABLE_DEFAULT = True
WV_LEVEL  = 2
WV_NBINS  = 2
WV_WAVELET = "db2"

# ---- NUEVO: filtro wavelet previo ----
PRE_WAVELET_FILTER_ENABLE = True  # True => filtrar 'precio' al principio
PRE_WV_REPLACE            = False    # True => reemplaza por la aproximación SWT; False => mezcla
PRE_WV_BLEND_ALPHA        = 0.6     # si REPLACE=False, y = alpha*approx + (1-alpha)*precio

# ---- NUEVO: simulación histórica rolling ----
ROLLING_HISTORICAL_SIM    = True   # True => Y1=real; Y2~train(Y1); Y3~train(Y1+Y2); ...

# ==================== Ploteo opcional ====================
PLOT_SUMMARY = True
LW_THIN = 0.9

import matplotlib
if PLOT_SUMMARY:
    try:
        current = matplotlib.get_backend().lower()
    except Exception:
        current = ""
    if "agg" in current:
        pass
    for bk in ("TkAgg","QtAgg","Qt5Agg","GTK3Agg","MacOSX","WXAgg"):
        try:
            matplotlib.use(bk, force=True); break
        except Exception:
            continue
else:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ==================== Libs científicas ====================
from scipy.stats import rankdata
from scipy.signal import welch
from scipy.interpolate import PchipInterpolator
from numpy.linalg import pinv

# PyWavelets (opcional)
try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False
    warnings.warn("PyWavelets no disponible: WV=True y el filtro previo caerán a suavizado por media móvil.")

# ==================== Utilidades de datos ====================
def load_and_resample(paths):
    dfs = []
    for p in paths:
        raw = pd.read_csv(p, sep="\t", encoding="latin1")
        dates = pd.to_datetime(raw["fecha"], format="%d/%m/%Y", dayfirst=True)
        hrs   = pd.to_numeric(raw["hora"], errors="coerce").fillna(0).astype(int).clip(0, 23)
        ts    = dates + pd.to_timedelta(hrs, unit="h")
        pr    = pd.to_numeric(raw["costo_en_dolares"].astype(str).str.replace(",", "."), errors="coerce")
        df = pd.DataFrame({"timestamp": ts, "precio": pr}).dropna()
        df = (df.set_index("timestamp").resample("h").mean().interpolate("time").reset_index())
        dfs.append(df)
    return pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)

def add_log_time(df, t0=None, use_offset=False):
    df["precio_log"] = np.log(df["precio"] + LOG_OFFSET_C) if use_offset else np.log1p(df["precio"])
    if t0 is None: t0 = df["timestamp"].iloc[0]
    df["t"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
    df["h"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    return t0

# ==================== SWT seguro / helpers ====================
def _safe_swt_level(n):
    if not HAVE_PYWT: return 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Lmax = pywt.swt_max_level(int(n))
    except Exception:
        Lmax = 0
    L = min(WV_LEVEL, int(Lmax))
    return max(0, L)

def _pad_to_multiple(x, m):
    if m <= 1: return x, 0
    n = len(x); r = n % m
    if r == 0: return x, 0
    pad = m - r
    xp = np.pad(x, (0, pad), mode="edge")
    return xp, pad

def swt_approx(x, wavelet=WV_WAVELET):
    x = np.asarray(x, float)
    if not HAVE_PYWT:
        # Suavizado fallback
        return pd.Series(x).rolling(window=24, min_periods=1, center=True).mean().values
    L = _safe_swt_level(len(x))
    if L <= 0:
        return pd.Series(x).rolling(window=24, min_periods=1, center=True).mean().values
    m = 2**L
    xp, pad = _pad_to_multiple(x, m)
    coeffs = pywt.swt(xp, wavelet=wavelet, level=L)
    cAs = [c[0] for c in coeffs]
    # Elige la escala con menor variación diferencial (tendencia)
    idx = int(np.argmin([np.var(np.diff(c)) if len(c)>1 else 0.0 for c in cAs]))
    approx = cAs[idx]
    if pad > 0: approx = approx[:-pad]
    return np.asarray(approx, float)

# ---- NUEVO: filtro wavelet previo a log/ajuste ----
def apply_pre_wavelet_filter(df):
    y = df["precio"].values.astype(float)
    approx = swt_approx(y)
    if PRE_WV_REPLACE:
        yp = approx
    else:
        a = float(PRE_WV_BLEND_ALPHA)
        yp = a*approx + (1.0 - a)*y
    yp = np.clip(yp, np.finfo(float).eps, np.inf)
    out = df.copy()
    out["precio"] = yp
    return out

# ==================== Modelado determinista / diseño ====================
def design_matrix(df, include_holidays=False):
    N = len(df)
    doy = df["timestamp"].dt.dayofyear.values
    L   = np.where(df["timestamp"].dt.is_leap_year, 366, 365)
    dow = df["timestamp"].dt.weekday.values
    weekend = (dow >= 5).astype(int)
    holiday = df["timestamp"].dt.date.isin(HOLIDAYS).astype(int) if include_holidays else np.zeros(N, int)
    phi_year = 2*np.pi*(doy - 1)/L
    phi_week = 2*np.pi*dow/7
    h = df["h"].values
    phi = [2*np.pi*k*h/24 for k in (1,2,3)]
    return np.column_stack([
        np.ones(N), df["t"].values,
        np.cos(phi_year), np.sin(phi_year),
        np.cos(phi_week), np.sin(phi_week),
        weekend, holiday,
        np.cos(phi[0]), np.sin(phi[0]),
        np.cos(phi[1]), np.sin(phi[1]),
        np.cos(phi[2]), np.sin(phi[2]),
    ])

def fit_with_cov(df, include_holidays=True, alpha_reg=1e-2):
    X = design_matrix(df, include_holidays)
    y = df["precio_log"].values
    XtX  = X.T @ X + alpha_reg * np.eye(X.shape[1])
    coef = np.linalg.solve(XtX, X.T @ y)
    covb = pinv(XtX)
    resid = y - X @ coef
    dof = max(1, len(y) - X.shape[1])
    sigma2 = np.sum(resid**2) / dof
    return coef, covb * sigma2, resid

# ==================== Hetero/PSD/Qmap ====================
def simulate_beta(coef, covb, rng, jitter=1e-10):
    k = covb.shape[0]
    covb_j = covb + jitter*np.eye(k)
    try:
        return rng.multivariate_normal(coef, covb_j)
    except Exception:
        return coef + rng.standard_normal(k)*np.sqrt(np.clip(np.diag(covb_j), 0, np.inf))

def hourly_std_from_hist(df_hist, resid_hist):
    s = (pd.Series(resid_hist, index=df_hist["timestamp"])
           .groupby(df_hist["timestamp"].dt.hour).std())
    fallback = float(np.std(resid_hist)) if np.std(resid_hist) > 0 else 1.0
    return s.reindex(range(24), fill_value=fallback).values

def hetero_apply_to_hours(r0, hours_seq, std_by_hour):
    r = r0 * std_by_hour[np.asarray(hours_seq, dtype=int)]
    s_in, s_out = np.std(r0), np.std(r)
    if s_out > 0 and s_in > 0:
        r *= (s_in / s_out)
    return r

def welch_psd(x):
    nper = min(1024, len(x))
    return welch(x, fs=1.0, nperseg=nper)

def iterative_psd(r0, resid_hist, use_clip=True, use_conv=False):
    f_hist, P_hist = welch_psd(resid_hist)
    N = len(r0)
    prev_err = np.inf
    for _ in range(PSD_MAX_ITER):
        f_s, P_s = welch_psd(r0)
        f0   = np.fft.rfftfreq(N, d=1.0)
        P_tg = np.interp(f0, f_hist, P_hist, left=P_hist[0], right=P_hist[-1])
        P_cr = np.interp(f0, f_s,   P_s,    left=P_s[0],    right=P_s[-1])
        ratio = np.sqrt(P_tg / (P_cr + 1e-12))
        if use_clip: ratio = np.clip(ratio, PSD_CLIP_RANGE[0], PSD_CLIP_RANGE[1])
        r0 = np.fft.irfft(np.fft.rfft(r0)*ratio, n=N)
        if use_conv:
            err = np.trapezoid(np.abs(P_cr - P_tg), f0)
            if abs(prev_err - err) < PSD_CONV_TOL:
                break
            prev_err = err
    return r0

def qmap_pchip(u, alphas, q_map):
    u = np.clip(u, alphas[0], alphas[-1])
    return PchipInterpolator(alphas, q_map, extrapolate=True)(u)

def quantile_mapping_global(r0, alphas, q_map, mean_shift=0.0):
    u = rankdata(r0, method="average") / (r0.size + 1.0)
    return qmap_pchip(u, alphas, q_map) + mean_shift

def build_wavelet_qmaps(resid_hist, ylog_hist, alphas, n_bins=WV_NBINS):
    if not HAVE_PYWT:
        return None, None, np.quantile(resid_hist, alphas)
    approx = swt_approx(ylog_hist)
    edges = np.quantile(approx, np.linspace(0, 1, n_bins+1))
    qmaps = {}
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (approx >= edges[b]) & (approx <= edges[b+1])
        else:
            mask = (approx >= edges[b]) & (approx <  edges[b+1])
        qmaps[b] = (np.quantile(resid_hist[mask], alphas) if np.sum(mask) >= 50 else None)
    q_global = np.quantile(resid_hist, alphas)
    return edges, qmaps, q_global

def wavelet_conditioned_mapping(r0, Ydet_log_sim, alphas, edges, qmaps_by_bin, q_global, mean_shift=0.0):
    N = r0.size
    u = rankdata(r0, method="average") / (N + 1.0)
    approx_sim = swt_approx(Ydet_log_sim) if HAVE_PYWT else None
    if approx_sim is None or edges is None or qmaps_by_bin is None:
        return qmap_pchip(u, alphas, q_global) + mean_shift
    bins = np.searchsorted(edges, approx_sim, side="right") - 1
    bins = np.clip(bins, 0, len(edges)-2)
    out = np.empty(N, float)
    for b in range(len(edges)-1):
        idx = np.where(bins == b)[0]
        if idx.size == 0: continue
        qmap = qmaps_by_bin.get(b, None)
        qvec = q_global if (qmap is None) else qmap
        out[idx] = qmap_pchip(u[idx], alphas, qvec)
    return out + mean_shift

# ==================== Simulación de un tramo ====================
def simulate_one(config, seed, X_val, coef, covb,
                 df_hist, resid_hist, alphas, hours_val,
                 wv_edges, wv_qmaps, q_global):
    rng = np.random.default_rng(int(seed))
    rc, hl, psd, clip, conv, wv = config["rc"], config["hl"], config["psd"], config["clip"], config["conv"], config["wv"]

    beta = simulate_beta(coef, covb, rng)
    Y_det_sim = X_val @ beta

    N = X_val.shape[0]
    base_std = np.std(resid_hist) if np.std(resid_hist) > 0 else 1.0
    r0 = rng.standard_normal(N) * base_std

    if hl:
        std_by_hour = hourly_std_from_hist(df_hist, resid_hist)
        r0 = hetero_apply_to_hours(r0, hours_val, std_by_hour)

    if psd:
        r0 = iterative_psd(r0, resid_hist, use_clip=clip, use_conv=conv)

    mean_shift = float(np.mean(resid_hist)) if rc else 0.0
    if wv and HAVE_PYWT and (wv_edges is not None):
        resid_sim = wavelet_conditioned_mapping(r0, Y_det_sim, alphas, wv_edges, wv_qmaps, q_global, mean_shift)
    else:
        resid_sim = quantile_mapping_global(r0, alphas, q_global, mean_shift)

    sim = np.expm1(Y_det_sim + resid_sim)
    return sim.astype(float), resid_sim.astype(float), Y_det_sim.astype(float)

# ==================== Métricas de selección ====================
def _robust_norm(v):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    q25, q75 = np.nanpercentile(v, 25), np.nanpercentile(v, 75)
    iqr = q75 - q25
    scale = iqr if iqr > 0 else (np.nanstd(v) if np.nanstd(v) > 0 else 1.0)
    return (v - med) / scale

def _acf_safe(x, nlags=72):
    x = np.array(x, dtype=float, copy=True)
    if x.size == 0:
        return np.zeros(nlags+1)
    x -= x.mean()
    if np.allclose(x.var(), 0):
        return np.zeros(nlags+1)
    c = np.correlate(x, x, mode='full')
    mid = c.size // 2
    ac = c[mid:mid+nlags+1]
    return ac / ac[0]

def _acf_distance(a, b, nlags=72):
    ra = _acf_safe(a, nlags); rb = _acf_safe(b, nlags)
    return float(np.sqrt(np.mean((ra - rb)**2)))

def psd_distance(real, sim):
    f_r, P_r = welch_psd(real); f_s, P_s = welch_psd(sim)
    if not np.array_equal(f_r, f_s):
        P_s = np.interp(f_r, f_s, P_s)
    return float(np.trapezoid((P_s - P_r)**2, f_r))

def welch_psd(x):
    nper = min(1024, len(x))
    return welch(x, fs=1.0, nperseg=nper)

# ==================== Guardado TXT ====================
def _config_true_tag(cfg_dict):
    order = ["rc","hl","psd","clip","conv","wv"]
    on = [k.upper() for k in order if cfg_dict.get(k, False)]
    return "_".join(on) if on else "NONE"

def _fmt3(x):
    try:
        x = float(x)
        if x == 0: return "0"
        p = int(np.floor(np.log10(abs(x))))
        return f"{x:.{max(0, 2 - p)}f}" if -3 <= p <= 2 else f"{x:.3g}"
    except Exception:
        return str(x)

def _write_sim_txt(path, timestamps, sim, real, header_lines):
    df_out = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
        "sim": np.asarray(sim, float),
        "real": np.asarray(real, float)
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write("# columns: timestamp\tsim\treal\n")
        df_out.to_csv(f, sep="\t", index=False)

def save_last_simulation_txts(timestamps_all, sim_total, real_all,
                              timestamps_val, sim_val, real_val,
                              cfg, best_idx, weights, sel_year_metrics,
                              seeds=None,
                              directory=SIM_SAVE_DIR, prefix=SIM_SAVE_PREFIX):
    os.makedirs(directory, exist_ok=True)
    tag = _config_true_tag(cfg)
    # TOTAL
    fname_total = f"{prefix}_{tag}_best{best_idx:03d}.txt"
    path_total  = os.path.join(directory, fname_total)
    mae_total  = float(np.mean(np.abs(sim_total - real_all)))
    hdr_total = [
        "LAST SIMULATION (TOTAL: histórico+validación)",
        f"config_true: {tag}",
        f"best_idx: {int(best_idx)}",
        (f"base_seed_best: {int(seeds[int(best_idx)])}" if (seeds is not None and len(seeds)>int(best_idx)) else ""),
        f"mae_total: {_fmt3(mae_total)}",
        f"weights(MAE,PSD,ACF): ({weights['mae']},{weights['psd']},{weights['acf']})",
        f"selection_year: {sel_year_metrics['year']}",
        f"selection_metrics (best) -> MAE={_fmt3(sel_year_metrics['mae'])}, PSD={_fmt3(sel_year_metrics['psd'])}, ACF={_fmt3(sel_year_metrics['acf'])}",
        f"rolling_hist_sim: {ROLLING_HISTORICAL_SIM}",
        f"pre_wv_filter: {PRE_WAVELET_FILTER_ENABLE}"
    ]
    _write_sim_txt(path_total, timestamps_all, sim_total, real_all, [h for h in hdr_total if h])
    # VALIDACIÓN
    fname_val = f"{prefix}_VALONLY_{tag}_best{best_idx:03d}.txt"
    path_val  = os.path.join(directory, fname_val)
    mae_val  = float(np.mean(np.abs(sim_val - real_val)))
    hdr_val = [
        "LAST SIMULATION (SOLO VALIDACIÓN)",
        f"config_true: {tag}",
        f"best_idx: {int(best_idx)}",
        (f"base_seed_best: {int(seeds[int(best_idx)])}" if (seeds is not None and len(seeds)>int(best_idx)) else ""),
        f"mae_val: {_fmt3(mae_val)}",
        f"weights(MAE,PSD,ACF): ({weights['mae']},{weights['psd']},{weights['acf']})",
        f"selection_year: {sel_year_metrics['year']}",
        f"selection_metrics (best) -> MAE={_fmt3(sel_year_metrics['mae'])}, PSD={_fmt3(sel_year_metrics['psd'])}, ACF={_fmt3(sel_year_metrics['acf'])}",
        f"rolling_hist_sim: {ROLLING_HISTORICAL_SIM}",
        f"pre_wv_filter: {PRE_WAVELET_FILTER_ENABLE}"
    ]
    _write_sim_txt(path_val, timestamps_val, sim_val, real_val, [h for h in hdr_val if h])

    print(f"[SAVE] TOTAL: {path_total}")
    print(f"[SAVE] VAL  : {path_val}")
    return path_total, path_val

# ==================== MAIN ====================
if __name__ == "__main__":
    # ---- 1) Datos ----
    df_hist = load_and_resample(TRAIN_PATHS)
    df_val  = load_and_resample([VAL_PATH])

    # ---- 1b) Filtro wavelet previo (opcional) ----
    if PRE_WAVELET_FILTER_ENABLE:
        df_hist = apply_pre_wavelet_filter(df_hist)
        df_val  = apply_pre_wavelet_filter(df_val)

    # ---- 2) Tiempo/log ----
    t0 = add_log_time(df_hist, use_offset=True)
    add_log_time(df_val, t0, use_offset=True)

    # ---- 3) Ajuste determinista GLOBAL (para VALIDACIÓN) ----
    #     (Siempre ajustamos con TODO el TRAIN para simular la VALIDACIÓN)
    coef_g, covb_g, resid_hist_global = fit_with_cov(df_hist, include_holidays=True, alpha_reg=1e-2)

    # ---- 4) Q-maps/wavelet GLOBAL (para VALIDACIÓN) ----
    alphas = np.linspace(0.001, 0.999, 999)
    if WV_ENABLE_DEFAULT and HAVE_PYWT:
        wv_edges_g, wv_qmaps_g, q_global_g = build_wavelet_qmaps(
            resid_hist_global, df_hist["precio_log"].values, alphas, n_bins=WV_NBINS)
    else:
        wv_edges_g, wv_qmaps_g, q_global_g = (None, None, np.quantile(resid_hist_global, alphas))

    # ---- 5) Features completos ----
    X_hist     = design_matrix(df_hist, include_holidays=True)
    X_val      = design_matrix(df_val, include_holidays=True)
    timestamps_hist = df_hist["timestamp"].values
    timestamps_val  = df_val["timestamp"].values
    hours_hist = df_hist["timestamp"].dt.hour.values
    hours_val  = df_val["timestamp"].dt.hour.values
    real_hist  = df_hist["precio"].values
    real_val   = df_val["precio"].values

    # ---- 6) Ensemble ----
    cfg = SINGLE_CONFIG.copy()
    base_ss = np.random.SeedSequence(4321)
    seeds = base_ss.generate_state(ENSEMBLE_SIZE, dtype=np.uint32).tolist()

    # ------- 6a) Simulación HISTÓRICA -------
    if ROLLING_HISTORICAL_SIM:
        hist_years = sorted(df_hist["timestamp"].dt.year.unique().tolist())
        hist_by_year = {y: df_hist[df_hist["timestamp"].dt.year == y].reset_index(drop=True) for y in hist_years}
        X_by_year_hist = {y: design_matrix(hist_by_year[y], include_holidays=True) for y in hist_years}
        H_by_year_hist = {y: hist_by_year[y]["timestamp"].dt.hour.values for y in hist_years}

        sims_hist = []
        for s in seeds:
            blocks = []
            past_df = None
            for idx, y in enumerate(hist_years):
                if idx == 0:
                    # primer año -> usa su propio real como "simulación"
                    blocks.append(hist_by_year[y]["precio"].values.copy())
                    past_df = hist_by_year[y].copy()
                    continue
                # Entrena con todo el pasado (hasta y-1)
                train_df = past_df.copy()
                c, cv, r = fit_with_cov(train_df, include_holidays=True, alpha_reg=1e-2)
                if WV_ENABLE_DEFAULT and HAVE_PYWT:
                    we, wq, qg = build_wavelet_qmaps(r, train_df["precio_log"].values, alphas, n_bins=WV_NBINS)
                else:
                    we, wq, qg = (None, None, np.quantile(r, alphas))
                Xy = X_by_year_hist[y]; Hy = H_by_year_hist[y]
                seed_y = int((int(s) + 10007 * (idx+1)) % (2**32-1))
                sim_b, _, _ = simulate_one(
                    cfg, seed_y, Xy, c, cv,
                    df_hist=train_df, resid_hist=r, alphas=alphas, hours_val=Hy,
                    wv_edges=we, wv_qmaps=wq, q_global=qg
                )
                blocks.append(sim_b)
                past_df = pd.concat([past_df, hist_by_year[y]], ignore_index=True)
            sims_hist.append(np.concatenate(blocks))
        sims_hist = np.stack(sims_hist, axis=0)
    else:
        # Modo global (igual que antes)
        sims_hist = []
        for s in seeds:
            sim_h, _, _ = simulate_one(
                cfg, int(s), X_hist, coef_g, covb_g,
                df_hist=df_hist, resid_hist=resid_hist_global, alphas=alphas, hours_val=hours_hist,
                wv_edges=wv_edges_g, wv_qmaps=wv_qmaps_g, q_global=q_global_g
            )
            sims_hist.append(sim_h)
        sims_hist = np.stack(sims_hist, axis=0)

    # ------- 6b) Simulación de VALIDACIÓN (siempre con TODO el TRAIN) -------
    sims_val = []
    for s in seeds:
        sim_v, _, _ = simulate_one(
            cfg, (int(s)+1000003)%(2**32-1), X_val, coef_g, covb_g,
            df_hist=df_hist, resid_hist=resid_hist_global, alphas=alphas, hours_val=hours_val,
            wv_edges=wv_edges_g, wv_qmaps=wv_qmaps_g, q_global=q_global_g
        )
        sims_val.append(sim_v)
    sims_val = np.stack(sims_val, axis=0)

    # ---- 7) SELECCIÓN en el AÑO ANTERIOR A VALIDACIÓN ----
    val_years = sorted(df_val["timestamp"].dt.year.unique().tolist())
    year_for_selection = (min(val_years) - 1)
    hist_years_all = sorted(df_hist["timestamp"].dt.year.unique().tolist())
    if year_for_selection not in hist_years_all:
        year_for_selection = max(hist_years_all)  # fallback
    years_hist_vec = pd.DatetimeIndex(timestamps_hist).year
    mask_sel = (years_hist_vec == year_for_selection)
    if mask_sel.sum() == 0:
        raise RuntimeError("No hay datos en TRAIN para el año de selección.")

    target_real = real_hist[mask_sel]
    candidate   = sims_hist[:, mask_sel]

    # Métricas por miembro (MAE, PSD, ACF)
    mae = np.mean(np.abs(candidate - target_real[None,:]), axis=1)
    psd = np.array([psd_distance(target_real, s) for s in candidate])
    acf = np.array([_acf_distance(target_real, s, nlags=72) for s in candidate])

    # Normalización robusta + pesos
    z_mae = _robust_norm(mae); z_psd = _robust_norm(psd); z_acf = _robust_norm(acf)
    w = dict(SELECT_WEIGHTS)
    w_mae = max(0.0, float(w.get("mae", 0.0)))
    w_psd = max(0.0, float(w.get("psd", 0.0)))
    w_acf = max(0.0, float(w.get("acf", 0.0)))
    w_sum = w_mae + w_psd + w_acf
    if w_sum <= 0.0: w_mae, w_sum = 1.0, 1.0
    w_mae /= w_sum; w_psd /= w_sum; w_acf /= w_sum

    score = w_mae*z_mae + w_psd*z_psd + w_acf*z_acf
    best_idx = int(np.nanargmin(score))
    print(f"[SELECT] Año selección={year_for_selection} | "
          f"best_idx={best_idx} (MAE={mae[best_idx]:.4g}, PSD={psd[best_idx]:.4g}, ACF={acf[best_idx]:.4g})")

    # ---- 8) TOTAL y guardado ----
    sim_hist0 = sims_hist[best_idx]
    sim_val0  = sims_val[best_idx]
    sim_total = np.concatenate([sim_hist0, sim_val0])

    timestamps_all = np.concatenate([timestamps_hist, timestamps_val])
    real_all       = np.concatenate([real_hist, real_val])

    if SAVE_LAST_SIM:
        os.makedirs(SIM_SAVE_DIR, exist_ok=True)
        save_last_simulation_txts(
            timestamps_all=timestamps_all,
            sim_total=sim_total,
            real_all=real_all,
            timestamps_val=timestamps_val,
            sim_val=sim_val0,
            real_val=real_val,
            cfg=cfg,
            best_idx=best_idx,
            weights=dict(mae=w_mae, psd=w_psd, acf=w_acf),
            sel_year_metrics=dict(year=year_for_selection,
                                  mae=float(mae[best_idx]),
                                  psd=float(psd[best_idx]),
                                  acf=float(acf[best_idx])),
            seeds=seeds,
            directory=SIM_SAVE_DIR,
            prefix=SIM_SAVE_PREFIX
        )

    # ---- 9) Plots resumen (opcional) ----
    if PLOT_SUMMARY:
        sims_total = np.concatenate([sims_hist, sims_val], axis=1)
        low_total, high_total = np.percentile(sims_total, [5,95], axis=0)
        mean_total = sims_total.mean(axis=0)

        plt.figure(figsize=(12,4))
        plt.fill_between(timestamps_all, low_total, high_total, alpha=0.3, label="5–95 %", color="C0")
        plt.plot(timestamps_all, mean_total, "C0-", linewidth=LW_THIN, label="Media simulaciones")
        plt.plot(timestamps_all, real_all, "k", linewidth=LW_THIN, label="Real")
        plt.xlabel("Fecha"); plt.ylabel("Precio")
        plt.title(f"TOTAL (Hist+Val) — mejor idx={best_idx} | selección en {year_for_selection} | "
                  f"rolling={ROLLING_HISTORICAL_SIM} | pre_wv={PRE_WAVELET_FILTER_ENABLE}")
        plt.legend(); plt.grid(True, linewidth=0.3); plt.tight_layout()
        try:
            os.makedirs(SIM_SAVE_DIR, exist_ok=True)
            plt.savefig(os.path.join(SIM_SAVE_DIR, f"{SIM_SAVE_PREFIX}_TOTAL_band_best{best_idx:03d}.png"), dpi=150)
        except Exception as e:
            print(f"[PLOT] No se pudo guardar la gráfica TOTAL: {e}")
        plt.show()

        # PSD del año de selección (real vs sim elegida)
        f_r, P_r = welch_psd(target_real); f_s, P_s = welch_psd(candidate[best_idx])
        plt.figure(figsize=(6,4))
        plt.semilogy(f_r, P_r, "k", linewidth=LW_THIN, label="Real (año selección)")
        plt.semilogy(f_s, P_s, "C0", linewidth=LW_THIN, label="Sim elegida")
        plt.xlabel("Frecuencia (1/h)"); plt.ylabel("PSD")
        plt.title("PSD en año de selección")
        plt.legend(); plt.grid(True, linewidth=0.3); plt.tight_layout(); plt.show()

    print("[DONE]")
