from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def _load_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return pd.DataFrame(), False
    return pd.read_csv(path), True


player_stats, HAS_PLAYER_STATS = _load_csv("mi_kkr_player_stats.csv")
vs_stats, HAS_VS_STATS = _load_csv("mi_kkr_vs_stats.csv")
main_df, HAS_MAIN_DF = _load_csv("mi_kkr_dynamic_dataset.csv")


def norm(x):
    return str(x).lower().strip()


def normalize_venue_name(venue):
    value = norm(venue)
    value = value.replace(",", " ")
    tokens = [token for token in value.split() if token not in {"stadium", "cricket", "ground"}]
    return " ".join(tokens)


def get_player_form(name):
    if not HAS_PLAYER_STATS or player_stats.empty:
        return None
    df = player_stats.copy()
    df["player"] = df["player"].astype(str).str.lower().str.strip()

    row = df[df["player"] == norm(name)]
    return float(row.iloc[0]["strike_rate"]) if len(row) else None


def get_vs(batsman, bowler):
    if not HAS_VS_STATS or vs_stats.empty:
        return None, None
    df = vs_stats.copy()
    df["batsman"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler"] = df["bowler"].astype(str).str.lower().str.strip()

    row = df[
        (df["batsman"] == norm(batsman)) &
        (df["bowler"] == norm(bowler))
    ]

    if len(row):
        return float(row.iloc[0]["strike_rate"]), float(row.iloc[0]["average"])

    return None, None


def get_venue_avg(venue):
    if not HAS_MAIN_DF or main_df.empty:
        return 160.0
    df = main_df.copy()
    df["venue_norm"] = df["venue"].astype(str).apply(normalize_venue_name)

    row = df[df["venue_norm"] == normalize_venue_name(venue)]
    if len(row):
        return float(row["current_score"].mean())

    return 160.0


def get_future_runs_column(predict_overs):
    over_value = int(max(round(float(predict_overs)), 1))
    if over_value <= 6:
        return "future_runs_6", 6
    if over_value <= 12:
        return "future_runs_12", 12
    return "future_runs_18", 18


def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def retrieve_similar_scenarios(score, wickets, over, balls, target_runs, predict_overs, striker, bowler, venue, limit=8):
    if not HAS_MAIN_DF or main_df.empty:
        return {
            "predicted_runs": None,
            "success_rate": None,
            "examples": [],
            "future_col": None,
            "exact_batsman_bowler_count": 0,
            "exact_venue_count": 0,
        }
    df = main_df.copy()

    query_current_rr = score / max(over + balls / 6.0, 0.1)
    query_required_rr = target_runs / max(float(predict_overs), 1.0)
    venue_norm = normalize_venue_name(venue)
    striker_norm = norm(striker)
    bowler_norm = norm(bowler)

    df["venue_norm"] = df["venue"].astype(str).apply(normalize_venue_name)
    df["batsman_norm"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler_norm"] = df["bowler"].astype(str).str.lower().str.strip()

    df["over_gap"] = (pd.to_numeric(df["over"], errors="coerce").fillna(0) - float(over)).abs()
    df["ball_gap"] = (pd.to_numeric(df["ball"], errors="coerce").fillna(0) - float(balls)).abs()
    df["score_gap"] = (pd.to_numeric(df["current_score"], errors="coerce").fillna(0) - float(score)).abs()
    df["wicket_gap"] = (pd.to_numeric(df["wickets"], errors="coerce").fillna(0) - float(wickets)).abs()
    df["rr_gap"] = (pd.to_numeric(df["current_rr"], errors="coerce").fillna(query_current_rr) - float(query_current_rr)).abs()
    df["required_rr_gap"] = (
        (pd.to_numeric(df["future_runs_6"], errors="coerce").fillna(0) / 6.0) - float(query_required_rr)
    ).abs()

    df["similarity_score"] = (
        df["over_gap"] * 6.0 +
        df["ball_gap"] * 1.5 +
        df["score_gap"] * 0.35 +
        df["wicket_gap"] * 10.0 +
        df["rr_gap"] * 9.0 +
        df["required_rr_gap"] * 12.0
    )

    df.loc[df["venue_norm"] == venue_norm, "similarity_score"] -= 8.0
    df.loc[df["batsman_norm"] == striker_norm, "similarity_score"] -= 10.0
    df.loc[df["bowler_norm"] == bowler_norm, "similarity_score"] -= 7.0

    future_col, supported_window = get_future_runs_column(predict_overs)
    df["window_runs"] = pd.to_numeric(df[future_col], errors="coerce")
    if int(max(round(float(predict_overs)), 1)) != supported_window:
        df["window_runs"] = df["window_runs"] * (float(predict_overs) / supported_window)

    top = df.dropna(subset=["window_runs"]).sort_values("similarity_score").head(limit).copy()

    if top.empty:
        return {
            "predicted_runs": None,
            "success_rate": None,
            "examples": [],
            "future_col": future_col
        }

    top["success"] = top["window_runs"] >= float(target_runs)

    examples = []
    for _, row in top.head(5).iterrows():
        examples.append({
            "venue": str(row.get("venue", "")),
            "batsman": str(row.get("batsman", "")),
            "bowler": str(row.get("bowler", "")),
            "score": int(_to_float(row.get("current_score"), 0)),
            "wickets": int(_to_float(row.get("wickets"), 0)),
            "over": int(_to_float(row.get("over"), 0)),
            "ball": int(_to_float(row.get("ball"), 0)),
            "window_runs": round(_to_float(row.get("window_runs"), 0), 1),
            "success": bool(row.get("success", False))
        })

    return {
        "predicted_runs": float(top["window_runs"].mean()),
        "success_rate": float(top["success"].mean()),
        "examples": examples,
        "future_col": future_col,
        "exact_batsman_bowler_count": int(
            len(top[(top["batsman_norm"] == striker_norm) & (top["bowler_norm"] == bowler_norm)])
        ),
        "exact_venue_count": int(len(top[top["venue_norm"] == venue_norm]))
    }


def build_historical_evidence(striker, bowler, venue, target_runs, predict_overs):
    if not HAS_MAIN_DF or main_df.empty:
        return []
    df = main_df.copy()
    df["venue_norm"] = df["venue"].astype(str).apply(normalize_venue_name)
    df["batsman_norm"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler_norm"] = df["bowler"].astype(str).str.lower().str.strip()

    future_col, supported_window = get_future_runs_column(predict_overs)
    df["window_runs"] = pd.to_numeric(df[future_col], errors="coerce")
    if int(max(round(float(predict_overs)), 1)) != supported_window:
        df["window_runs"] = df["window_runs"] * (float(predict_overs) / supported_window)

    venue_norm = normalize_venue_name(venue)
    striker_norm = norm(striker)
    bowler_norm = norm(bowler)

    def summarize(mask, label):
        subset = df[mask].dropna(subset=["window_runs"])
        total = len(subset)
        if total == 0:
            return None
        success = int((subset["window_runs"] >= float(target_runs)).sum())
        avg_runs = float(subset["window_runs"].mean())
        return (
            f"{success} out of {total} {label} reached {target_runs}+ runs in the next "
            f"{predict_overs} overs. Average return in that window: {avg_runs:.1f} runs."
        )

    evidence = []
    for mask, label, priority in [
        ((df["batsman_norm"] == striker_norm) & (df["venue_norm"] == venue_norm),
         f"{striker} at {venue} situations", 1),
        ((df["batsman_norm"] == striker_norm) & (df["bowler_norm"] == bowler_norm),
         f"{striker} vs {bowler} situations", 0),
        ((df["venue_norm"] == venue_norm),
         f"{venue} situations", 2),
    ]:
        summary = summarize(mask, label)
        if summary:
            evidence.append((priority, summary))

    evidence.sort(key=lambda item: item[0])
    return [summary for _, summary in evidence]
