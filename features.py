import numpy as np
import pandas as pd

DEFAULT_STATS = ["pts", "opp_pts", "q1", "q2", "q3", "q4", "ot", "reb", "ast", "stl", "ortg"]
DEFAULT_CTX = ["days_rest", "distance", "pts_r10", "opp_pts_r10", "pts_r20", "opp_pts_r20"]
WINDOWS = (10, 20)
LAGS = 3

def build_dataset(logs, odds):
    STATS = [c for c in DEFAULT_STATS if c in logs.columns]
    logs = logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    odds = odds.copy()
    odds["date"] = pd.to_datetime(odds["date"])
    logs = logs.sort_values(["team", "date"]).reset_index(drop=True)
    avg = logs.groupby(["season", "team"])[STATS].transform(lambda s: s.expanding().mean().shift(1))
    for c in STATS:
        logs["s_" + c] = avg[c]
    opp = logs[["date", "team"] + ["s_" + c for c in STATS]].rename(
        columns={"team": "opp", **{"s_" + c: "o_" + c for c in STATS}})
    logs = logs.merge(opp, on=["date", "opp"], how="left")
    feat = STATS + ["o_" + c for c in STATS]
    logs[feat] = logs[feat].fillna(logs[feat].mean())
    logs = logs.sort_values(["team", "date"]).reset_index(drop=True)
    g = logs.groupby("team")
    for w in WINDOWS:
        for c in ("pts", "opp_pts"):
            logs[f"{c}_r{w}"] = g[c].transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=w).mean())
    CTX = [c for c in DEFAULT_CTX if c in logs.columns]
    for k in range(1, LAGS + 1):
        lag = g[feat].shift(k)
        for c in feat:
            logs[f"{c}_l{k}"] = lag[c]
    lagcols = [f"{c}_l{k}" for k in range(1, LAGS + 1) for c in feat]
    logs = logs.dropna(subset=lagcols + [c for c in CTX if c.endswith("_r10") or c.endswith("_r20")])
    home = logs[logs["is_home"] == 1]
    away = logs[logs["is_home"] == 0]
    m = home.merge(away, left_on=["date", "team"], right_on=["date", "opp"], suffixes=("_h", "_a"))
    m = m.merge(odds, left_on=["date", "team_h"], right_on=["date", "home"], how="inner")
    seq = []
    for k in range(LAGS, 0, -1):
        seq.append(m[[f"{c}_l{k}_h" for c in feat] + [f"{c}_l{k}_a" for c in feat]].to_numpy(dtype=np.float32))
    seq = np.stack(seq, axis=1)
    ctx = m[[c + "_h" for c in CTX] + [c + "_a" for c in CTX]].to_numpy(dtype=np.float32)
    y = (m["pts_h"] + m["pts_a"]).to_numpy(dtype=np.float32)
    return dict(seq=seq, ctx=ctx, y=y, line=m["ou_line"].to_numpy(dtype=np.float32),
                season=m["season_h"].to_numpy(), date=m["date"].to_numpy(),
                home=m["team_h"].to_numpy(), away=m["team_a"].to_numpy())

def split(data, train, val, test):
    out = {}
    for name, seasons in (("train", train), ("val", val), ("test", test)):
        idx = np.isin(data["season"], seasons)
        out[name] = {k: v[idx] for k, v in data.items()}
    return out

def flat(d):
    return np.concatenate([d["seq"].reshape(len(d["y"]), -1), d["ctx"]], axis=1)
