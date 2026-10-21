import os
import urllib.request

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
RAW = "data/all_stocks_5yr.csv"

WINDOW = 20      # days of history fed to the model
HORIZON = 5      # predict the return over the next 5 trading days


def download():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(RAW):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)


def load_prices():
    download()
    df = pd.read_csv(RAW, parse_dates=["date"])
    df = df.dropna(subset=["close", "volume"])
    close = df.pivot(index="date", columns="Name", values="close")
    volume = df.pivot(index="date", columns="Name", values="volume")
    # keep names with a full history so the panel is rectangular
    full = close.columns[close.notna().mean() > 0.99]
    close = close[full].ffill()
    volume = volume[full].ffill()
    return close, volume


def build_features(close, volume):
    ret = np.log(close).diff()
    vol20 = ret.rolling(20).std()
    dvol = np.log(volume).diff()
    mom = np.log(close / close.shift(20))
    # cross sectionally standardise each feature each day so the model
    # sees relative not absolute values
    feats = {}
    for name, f in [("ret", ret), ("vol20", vol20), ("dvol", dvol), ("mom", mom)]:
        feats[name] = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1) + 1e-9, axis=0)
    fwd = np.log(close.shift(-HORIZON) / close)
    return feats, fwd, ret


def make_windows(feats, fwd):
    # returns X of shape (samples, WINDOW, n_features), y, plus date and ticker ids
    names = list(feats.keys())
    arr = np.stack([feats[n].values for n in names], axis=-1)  # (T, N, F)
    T, N, F = arr.shape
    dates = feats[names[0]].index
    tickers = feats[names[0]].columns
    X, y, d_idx, t_idx = [], [], [], []
    for t in range(WINDOW + 20, T - HORIZON):
        win = arr[t - WINDOW + 1:t + 1]          # (WINDOW, N, F)
        target = fwd.values[t]                   # (N,)
        ok = ~np.isnan(win).any(axis=(0, 2)) & ~np.isnan(target)
        X.append(win[:, ok, :].transpose(1, 0, 2))
        y.append(target[ok])
        d_idx.append(np.full(ok.sum(), t))
        t_idx.append(np.where(ok)[0])
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)
    return X, y, np.concatenate(d_idx), np.concatenate(t_idx), dates, tickers
