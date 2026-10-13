import sys
import pandas as pd

def prepare(src, out_dir="data"):
    df = pd.read_csv(src)
    df = df[df["regular"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["total"])
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "date": df["date"], "season": df["season"],
            "team": df[side], "opp": df[other],
            "is_home": int(side == "home"),
            "pts": df[f"score_{side}"], "opp_pts": df[f"score_{other}"],
            "q1": df[f"q1_{side}"], "q2": df[f"q2_{side}"],
            "q3": df[f"q3_{side}"], "q4": df[f"q4_{side}"],
            "ot": df[f"ot_{side}"],
        })
        rows.append(part)
    logs = pd.concat(rows).sort_values(["team", "date"]).reset_index(drop=True)
    logs["days_rest"] = logs.groupby("team")["date"].diff().dt.days.fillna(7).clip(upper=7)
    odds = pd.DataFrame({"date": df["date"], "home": df["home"], "away": df["away"], "ou_line": df["total"]})
    logs.to_csv(f"{out_dir}/logs.csv", index=False)
    odds.to_csv(f"{out_dir}/odds.csv", index=False)
    print(f"logs {len(logs)} rows, odds {len(odds)} rows, seasons {sorted(df['season'].unique())}")

if __name__ == "__main__":
    prepare(sys.argv[1])
