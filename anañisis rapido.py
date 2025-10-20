import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import welch

# -----------------------------------------------------------------------
# 1) Ficheros de entrada: primeros 4 años para entrenar, último año para validar
paths = [
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2018.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2019.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2020.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2021.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2022.tsv.TXT",
    r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2023.tsv.TXT",
]
path_val = r"E:\analisi_datos_entrada\DATOS\BRASIL\BA02T0002SE032T0002_2024.tsv.TXT"

# -----------------------------------------------------------------------
# 2) Función de lectura
def load_df(path):
    raw = pd.read_csv(path, sep='\t', encoding='latin1')
    fechas = pd.to_datetime(raw['fecha'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    horas  = pd.to_numeric(raw['hora'], errors='coerce').fillna(0).astype(int)
    ts     = fechas + pd.to_timedelta(horas//24, 'D') + pd.to_timedelta(horas%24, 'h')
    precio = (raw['costo_en_dolares']
              .astype(str)
              .str.replace(',', '.', regex=False)
              .str.strip()
             ).pipe(pd.to_numeric, errors='coerce')
    return pd.DataFrame({'timestamp': ts, 'precio': precio}) \
             .dropna(subset=['timestamp','precio']) \
             .sort_values('timestamp') \
             .reset_index(drop=True)

df_train = pd.concat([load_df(p) for p in paths], ignore_index=True)
df_val   = load_df(path_val)

# -----------------------------------------------------------------------
# 3) Log-transform y t en horas
c = df_train['precio'].max()
for df in (df_train, df_val):
    df['precio_log'] = np.log(df['precio'] + c)
    t0 = df_train['timestamp'].iloc[0]
    df['t'] = (df['timestamp'] - t0).dt.total_seconds() / 3600

# -----------------------------------------------------------------------
# 4) Ajuste parte determinista (un solo OLS, idéntico al de Julia)
def fit_deterministic_global(df_train):
    df = df_train
    # dummies de mes
    df['mes'] = df['timestamp'].dt.month
    D_mes = pd.get_dummies(df['mes'], prefix='m', drop_first=True)
    # anual
    df['d_ano'] = df['timestamp'].dt.dayofyear
    df['dias_ano'] = np.where(df['timestamp'].dt.is_leap_year, 366, 365)
    df['sa_cos'] = np.cos(2*np.pi*(df['d_ano']-1)/df['dias_ano'])
    df['sa_sin'] = np.sin(2*np.pi*(df['d_ano']-1)/df['dias_ano'])
    # semanal
    df['d_sem'] = df['timestamp'].dt.weekday
    df['sw_cos'] = np.cos(2*np.pi*df['d_sem']/7)
    df['sw_sin'] = np.sin(2*np.pi*df['d_sem']/7)
    # fines de semana
    df['wkend'] = df['d_sem'].isin([5,6]).astype(int)

    # construir X
    X = pd.concat([
        pd.Series(1, index=df.index, name='const'),
        df['t'].rename('t'),
        df['sa_cos'], df['sa_sin'],
        df['sw_cos'], df['sw_sin'],
        df['wkend'],
        D_mes
    ], axis=1)

    # convertir a float puro
    X_mat = X.astype(float).values
    y_vec = df['precio_log'].values

    # OLS
    beta = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ y_vec)
    # guardar predicción y residuales
    df['Ydet'] = X_mat @ beta
    df['resid'] = y_vec - df['Ydet']
    return beta, X.columns

beta, Xcols = fit_deterministic_global(df_train)

# -----------------------------------------------------------------------
# 5) Generar simulación espectral gaussiana
def generate_gaussian(resid, L):
    f, Pxx = welch(resid - resid.mean(),
                   fs=0.5, window='hann',
                   nperseg=L, noverlap=L//2,
                   nfft=L, return_onesided=True,
                   scaling='density')
    mag = np.sqrt(Pxx * 0.5)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2*np.pi, size=len(f))
    Xpos   = mag * np.exp(1j * phases)
    Xfull  = np.zeros(L, dtype=complex)
    h = len(Xpos)
    Xfull[:h] = Xpos
    Xfull[h:] = np.conj(Xpos[1:L-h+1][::-1])
    return np.fft.ifft(Xfull).real

Lval = len(df_val)
XG   = generate_gaussian(df_train['resid'].values, Lval)

# -----------------------------------------------------------------------
# 6) Mapeo cuantílico global
r_sorted = np.sort(df_train['resid'].values)
probs    = np.arange(1, len(r_sorted)+1) / len(r_sorted)
inv_cdf  = interp1d(probs, r_sorted, kind='linear',
                    bounds_error=False,
                    fill_value=(r_sorted[0], r_sorted[-1]))
uG  = norm.cdf((XG - XG.mean())/XG.std())
XNG = inv_cdf(uG)

# -----------------------------------------------------------------------
# 7) Reconstruir validación con el mismo OLS
def design_matrix_val(df_val, df_train, beta, Xcols):
    df = df_val.copy()
    df['mes'] = df['timestamp'].dt.month
    D_mes = pd.get_dummies(df['mes'], prefix='m', drop_first=True)
    # asegurar mismas columnas de mes
    for col in [c for c in Xcols if c.startswith('m_') and c not in D_mes]:
        D_mes[col] = 0
    df['d_ano']  = df['timestamp'].dt.dayofyear
    df['dias_ano']= np.where(df['timestamp'].dt.is_leap_year, 366, 365)
    df['sa_cos'] = np.cos(2*np.pi*(df['d_ano']-1)/df['dias_ano'])
    df['sa_sin'] = np.sin(2*np.pi*(df['d_ano']-1)/df['dias_ano'])
    df['d_sem']  = df['timestamp'].dt.weekday
    df['sw_cos'] = np.cos(2*np.pi*df['d_sem']/7)
    df['sw_sin'] = np.sin(2*np.pi*df['d_sem']/7)
    df['wkend']  = df['d_sem'].isin([5,6]).astype(int)

    # reconstruir X_val en el mismo orden de columnas Xcols
    X_val = pd.concat([
        pd.Series(1, index=df.index, name='const'),
        df['t'].rename('t'),
        df['sa_cos'], df['sa_sin'],
        df['sw_cos'], df['sw_sin'],
        df['wkend'],
        D_mes
    ], axis=1)[Xcols].astype(float).values

    return X_val @ beta

df_val['Ydet'] = design_matrix_val(df_val, df_train, beta, Xcols)
df_val['resid_sim'] = XNG
df_val['log_sim']   = df_val['Ydet'] + df_val['resid_sim']
df_val['precio_sim'] = np.exp(df_val['log_sim']) - c

# -----------------------------------------------------------------------
# 8) Gráficos de validación

# a) Real vs Sim
plt.figure(figsize=(12,4))
plt.plot(df_val['timestamp'], df_val['precio'],     label='Real')
plt.plot(df_val['timestamp'], df_val['precio_sim'], label='Simulado', alpha=0.8)
plt.title('Precio Real vs Simulado (2023)')
plt.xlabel('Fecha'), plt.ylabel('USD')
plt.legend(), plt.grid(True), plt.tight_layout()
plt.show()

# b) PDF & CDF precios
def ecdf(arr):
    x = np.sort(arr); y = np.arange(1,len(x)+1)/len(x)
    return x, y

xr, yr = ecdf(df_val['precio'])
xs, ys = ecdf(df_val['precio_sim'])
plt.figure(figsize=(14,4))
plt.subplot(1,2,1)
plt.hist(df_val['precio'],    bins=50, density=True, alpha=0.5, label='Real')
plt.hist(df_val['precio_sim'], bins=50, density=True, alpha=0.5, label='Sim')
plt.title('PDF Precios'), plt.xlabel('USD'), plt.legend()
plt.subplot(1,2,2)
plt.step(xr, yr, where='post', label='Real')
plt.step(xs, ys, where='post', label='Sim')
plt.title('CDF Precios'), plt.xlabel('USD'), plt.legend()
plt.tight_layout(), plt.show()

# c) ACF precios
def acf(x,n):
    x = x-x.mean(); return np.array([1]+[np.corrcoef(x[:-lag],x[lag:])[0,1]
                                       for lag in range(1,n+1)])
nl=24*15
acf_r = acf(df_val['precio'].values, nl)
acf_s = acf(df_val['precio_sim'].values, nl)
plt.figure(figsize=(10,4))
plt.plot(acf_r,label='Real'), plt.plot(acf_s, '--',label='Sim')
plt.title('ACF Precios'), plt.xlabel('Lag (h)'), plt.legend(), plt.grid(True), plt.tight_layout()
plt.show()

# d) Serie completa: histórico vs 2023 simulado
hist = df_train[['timestamp','precio']].assign(serie='Train')
sim  = df_val[['timestamp','precio_sim']].rename(columns={'precio_sim':'precio'}).assign(serie='Sim2023')
all_ = pd.concat([hist, sim], ignore_index=True)
plt.figure(figsize=(14,4))
for name,g in all_.groupby('serie'):
    plt.plot(g['timestamp'], g['precio'], label=name, alpha=0.8)
plt.title('Histórico vs 2023 Simulado'), plt.xlabel('Fecha'), plt.ylabel('USD')
plt.legend(), plt.grid(True), plt.tight_layout()
plt.show()
