import numpy as np
import pandas as pd

def make(seasons=range(2012, 2019), games_per_day=8, days=120, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(30)]
    off = rng.normal(0, 3, 30)
    dfn = rng.normal(0, 3, 30)
    rows, odds = [], []
    last = {t: None for t in teams}
    for s in seasons:
        drift = rng.normal(0, 1)
        start = pd.Timestamp(f"{s}-10-20")
        last = {t: None for t in teams}
        for d in range(days):
            date = start + pd.Timedelta(days=d)
            order = rng.permutation(30)[: 2 * games_per_day]
            for k in range(games_per_day):
                h, a = order[2 * k], order[2 * k + 1]
                mu_h = 100 + 0.35 * (s - 2012) + off[h] - dfn[a] + 2 + drift
                mu_a = 100 + 0.35 * (s - 2012) + off[a] - dfn[h] + drift
                ph, pa = rng.normal(mu_h, 9), rng.normal(mu_a, 9)
                line = mu_h + mu_a + rng.normal(0, 3)
                odds.append(dict(date=date, home=teams[h], away=teams[a], ou_line=round(line * 2) / 2))
                for me, op, p, q, ih in ((h, a, ph, pa, 1), (a, h, pa, ph, 0)):
                    t = teams[me]
                    rest = 3 if last[t] is None else (date - last[t]).days
                    last[t] = date
                    rows.append(dict(date=date, season=s, team=t, opp=teams[op], is_home=ih,
                                     pts=round(p), opp_pts=round(q),
                                     reb=rng.normal(44, 5), ast=rng.normal(23, 4), stl=rng.normal(7, 2),
                                     ortg=p + rng.normal(0, 4), days_rest=rest,
                                     distance=rng.uniform(0, 2500) * (1 - ih)))
    return pd.DataFrame(rows), pd.DataFrame(odds)

if __name__ == "__main__":
    logs, odds = make()
    logs.to_csv("data/logs.csv", index=False)
    odds.to_csv("data/odds.csv", index=False)
