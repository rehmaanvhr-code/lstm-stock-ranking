import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import data

SEED = 0
EPOCHS = 8
BATCH = 1024
LR = 1e-3
COST_BPS = 10          # one way transaction cost per rebalance
TRAIN_END = "2016-06-30"
VAL_END = "2016-12-31"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)


class LSTMRanker(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1])
        return self.head(h).squeeze(1)


def rank_ic(pred, y, d_idx):
    # spearman correlation between prediction and realised return, each day
    ics = []
    for d in np.unique(d_idx):
        m = d_idx == d
        if m.sum() > 10:
            ics.append(spearmanr(pred[m], y[m]).correlation)
    ics = np.array(ics)
    return ics.mean(), ics.mean() / (ics.std() / np.sqrt(len(ics))), ics


def backtest(pred, y, d_idx, t_idx, dates, n_tickers):
    # every HORIZON days go long the top decile and short the bottom decile,
    # equal weight, and pay COST_BPS on the fraction of the book that turned over
    days = np.unique(d_idx)
    rebalance = days[::data.HORIZON]
    gross, net, stamps = [], [], []
    w_old = np.zeros(n_tickers)
    for d in rebalance:
        m = d_idx == d
        p, r, t = pred[m], y[m], t_idx[m]
        k = max(1, len(p) // 10)
        order = np.argsort(p)
        w = np.zeros(n_tickers)
        w[t[order[-k:]]] = 1.0 / k
        w[t[order[:k]]] = -1.0 / k
        turnover = np.abs(w - w_old).sum()      # 4.0 at most when the whole book flips
        w_old = w
        g = r[order[-k:]].mean() - r[order[:k]].mean()
        gross.append(g)
        net.append(g - turnover * COST_BPS / 1e4)
        stamps.append(dates[d])
    idx = pd.DatetimeIndex(stamps)
    return pd.Series(gross, index=idx), pd.Series(net, index=idx)


def predict_torch(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(torch.tensor(X[i:i + 4096])).numpy())
    return np.concatenate(out)


def main():
    close, volume = data.load_prices()
    print("panel", close.shape[0], "days", close.shape[1], "tickers")
    feats, fwd, ret = data.build_features(close, volume)
    X, y, d_idx, t_idx, dates, tickers = data.make_windows(feats, fwd)
    print("samples", X.shape)

    sample_dates = dates[d_idx]
    tr = sample_dates <= pd.Timestamp(TRAIN_END)
    va = (sample_dates > pd.Timestamp(TRAIN_END)) & (sample_dates <= pd.Timestamp(VAL_END))
    te = sample_dates > pd.Timestamp(VAL_END)
    print("train %d  val %d  test %d" % (tr.sum(), va.sum(), te.sum()))

    # targets are cross sectionally standardised per day so the model learns ranking
    y_std = y.copy()
    for d in np.unique(d_idx):
        m = d_idx == d
        y_std[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-9)

    results = {}

    # baseline 1: last month momentum, just the final value of the mom feature
    mom_pred = X[:, -1, 3]
    results["momentum"] = rank_ic(mom_pred[te], y[te], d_idx[te])[:2]

    # baseline 2: ridge regression on the flattened window
    flat = X.reshape(len(X), -1)
    ridge = Ridge(alpha=100.0).fit(flat[tr], y_std[tr])
    ridge_pred = ridge.predict(flat)
    results["ridge"] = rank_ic(ridge_pred[te], y[te], d_idx[te])[:2]

    # the LSTM
    model = LSTMRanker(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y_std[tr])
    best_ic, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        va_ic = rank_ic(predict_torch(model, X[va]), y[va], d_idx[va])[0]
        print("epoch %d  train mse %.4f  val rank IC %.4f" % (ep + 1, total / len(Xtr), va_ic))
        if va_ic > best_ic:
            best_ic = va_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    lstm_pred = predict_torch(model, X)
    results["lstm"] = rank_ic(lstm_pred[te], y[te], d_idx[te])[:2]

    print("\ntest period rank IC (mean, t stat)")
    for k, v in results.items():
        print("  %-9s %.4f  t=%.2f" % (k, v[0], v[1]))

    curves = {}
    for name, p in [("momentum", mom_pred), ("ridge", ridge_pred), ("lstm", lstm_pred)]:
        g, n = backtest(p[te], y[te], d_idx[te], t_idx[te], dates, len(tickers))
        curves[name + " gross"] = g
        curves[name + " net"] = n
    # equal weight long only market over the same rebalance dates for reference
    mkt = []
    for d in np.unique(d_idx[te])[::data.HORIZON]:
        mkt.append(y[te][d_idx[te] == d].mean())
    curves["equal weight market"] = pd.Series(mkt, index=curves["lstm net"].index)

    summary = {}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in curves.items():
        ann_ret = s.mean() * 252 / data.HORIZON
        ann_vol = s.std() * np.sqrt(252 / data.HORIZON)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        summary[name] = {"annual_return": float(ann_ret), "annual_vol": float(ann_vol), "sharpe": float(sharpe)}
        style = ":" if name.endswith("gross") else "-"
        ax.plot(s.cumsum(), style, label="%s  sharpe %.2f" % (name, sharpe))
        print("  %-20s ann ret %6.1f%%  ann vol %5.1f%%  sharpe %5.2f" % (name, 100 * ann_ret, 100 * ann_vol, sharpe))
    ax.set_title("test period 2017 to 2018, long top decile short bottom decile, %d bps costs" % COST_BPS)
    ax.set_ylabel("cumulative log return")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/backtest.png", dpi=120)

    json.dump({"rank_ic": {k: {"mean": float(v[0]), "t_stat": float(v[1])} for k, v in results.items()},
               "backtest": summary,
               "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
              open("results/metrics.json", "w"), indent=2)
    torch.save(best_state, "results/lstm.pt")


if __name__ == "__main__":
    main()
