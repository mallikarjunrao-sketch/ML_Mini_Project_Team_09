import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Y_MEAN, Y_STD = 200.0, 20.0

def random_forest(Xtr, ytr, Xte):
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte)

def ridge(Xtr, ytr, Xte):
    mod = make_pipeline(StandardScaler(), Ridge(alpha=1000))
    mod.fit(Xtr, ytr)
    return mod.predict(Xte)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 500), nn.ReLU(), nn.Linear(500, 100), nn.ReLU(),
                                 nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 1))

    def forward(self, x, ctx=None):
        return self.net(x).squeeze(-1)

class LSTMNet(nn.Module):
    def __init__(self, d_in, d_ctx, hidden=20, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + d_ctx, 1)

    def forward(self, x, ctx):
        _, (h, _) = self.lstm(x)
        return self.out(torch.cat([self.drop(h[-1]), ctx], dim=1)).squeeze(-1)

def fit_torch(model, tr, va, epochs=80, lr=1e-3, wd=1e-4, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    Xtr, Ctr, ytr = [torch.tensor(a) for a in tr]
    Xva, Cva, yva = [torch.tensor(a) for a in va]
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[b], Ctr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva, Cva), yva).item()
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model

def predict_torch(model, X, C):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X), torch.tensor(C)).numpy() * Y_STD + Y_MEAN

def scaler(train_arr):
    flat_ = train_arr.reshape(-1, train_arr.shape[-1])
    mu, sd = flat_.mean(0), flat_.std(0) + 1e-6
    return lambda a: ((a - mu) / sd).astype(np.float32)

def neural_net(tr, va, te):
    sx = scaler(np.concatenate([tr["seq"].reshape(len(tr["y"]), -1), tr["ctx"]], 1))
    f = lambda d: sx(np.concatenate([d["seq"].reshape(len(d["y"]), -1), d["ctx"]], 1))
    ys = lambda d: ((d["y"] - Y_MEAN) / Y_STD).astype(np.float32)
    dummy = lambda d: np.zeros((len(d["y"]), 1), dtype=np.float32)
    model = MLP(f(tr).shape[1])
    model = fit_torch(model, (f(tr), dummy(tr), ys(tr)), (f(va), dummy(va), ys(va)), wd=1e-3)
    return predict_torch(model, f(te), dummy(te))

def lstm_net(tr, va, te):
    ss, sc = scaler(tr["seq"]), scaler(tr["ctx"])
    ys = lambda d: ((d["y"] - Y_MEAN) / Y_STD).astype(np.float32)
    pack = lambda d: (ss(d["seq"]), sc(d["ctx"]))
    model = LSTMNet(tr["seq"].shape[2], tr["ctx"].shape[1])
    model = fit_torch(model, (*pack(tr), ys(tr)), (*pack(va), ys(va)), epochs=60)
    return predict_torch(model, *pack(te))

def collaborative_filtering(all_games, te, window=5, rank=5, iters=30):
    teams = sorted(set(all_games["home"]) | set(all_games["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    all_games = all_games.assign(h=all_games["home"].map(ix), a=all_games["away"].map(ix))
    gm = all_games["total"].mean()
    preds = {}
    for d in np.unique(te["date"]):
        past = all_games[(all_games["date"] < d) & (all_games["date"] >= d - np.timedelta64(window, "D"))]
        if len(past) == 0:
            preds[d] = None
            continue
        A = np.zeros((n, n))
        cnt = np.zeros((n, n))
        np.add.at(A, (past["h"].to_numpy(), past["a"].to_numpy()), past["total"].to_numpy())
        np.add.at(cnt, (past["h"].to_numpy(), past["a"].to_numpy()), 1)
        mask = cnt > 0
        A[mask] /= cnt[mask]
        mu = A[mask].mean()
        Z = np.where(mask, A, mu)
        for _ in range(iters):
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            R = (U[:, :rank] * S[:rank]) @ Vt[:rank]
            Z = np.where(mask, A, R)
        preds[d] = R
    out = np.empty(len(te["y"]))
    for i, (d, h, a) in enumerate(zip(te["date"], te["home"], te["away"])):
        R = preds[d]
        out[i] = gm if R is None else R[ix[h], ix[a]]
    return out
