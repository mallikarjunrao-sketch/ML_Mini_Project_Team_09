import argparse
import numpy as np
import pandas as pd
from features import build_dataset, split, flat
import models

def beat_line(pred, y, line):
    keep = y != line
    return float(np.mean((pred[keep] > line[keep]) == (y[keep] > line[keep])))

def mse(p, y):
    return float(np.mean((p - y) ** 2))

def seasons(s):
    return [int(x) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/logs.csv")
    ap.add_argument("--odds", default="data/odds.csv")
    ap.add_argument("--train", type=seasons, default=[2013, 2014, 2015, 2016])
    ap.add_argument("--val", type=seasons, default=[2017])
    ap.add_argument("--test", type=seasons, default=[2018])
    a = ap.parse_args()

    data = build_dataset(pd.read_csv(a.logs), pd.read_csv(a.odds))
    sp = split(data, a.train, a.val, a.test)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"train {len(tr['y'])}  val {len(va['y'])}  test {len(te['y'])}")

    games = pd.DataFrame({"date": data["date"], "home": data["home"], "away": data["away"], "total": data["y"]})
    results = {
        "Ridge Regression": models.ridge(flat(tr), tr["y"], flat(te)),
        "Random Forest": models.random_forest(flat(tr), tr["y"], flat(te)),
        "Collaborative Filtering": models.collaborative_filtering(games, te),
        "Neural Network": models.neural_net(tr, va, te),
        "LSTM": models.lstm_net(tr, va, te),
        "Sportsbook line": te["line"],
    }
    rows = [{"Model": k, "Test MSE": round(mse(p, te["y"]), 2),
             "Beats line %": "-" if k == "Sportsbook line" else round(100 * beat_line(p, te["y"], te["line"]), 1)}
            for k, p in results.items()]
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
