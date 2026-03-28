import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = Path.home() / "Downloads"


def resolve_csv_path(filename):
    candidates = [
        DOWNLOADS_DIR / filename,
        BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return BASE_DIR / filename

player_stats = pd.read_csv(resolve_csv_path("rcb_srh_player_stats.csv"))
vs_stats = pd.read_csv(resolve_csv_path("rcb_srh_vs_stats.csv"))
main_df = pd.read_csv(resolve_csv_path("rcb_srh_final_dataset.csv"))

def norm(x):
    return str(x).lower().strip()

def get_player_form(name):
    df = player_stats.copy()
    df["batsman"] = df["batsman"].str.lower()

    row = df[df["batsman"] == norm(name)]
    return row.iloc[0]["avg"] if len(row) else 30

def get_vs(batsman, bowler):
    df = vs_stats.copy()
    df["batsman"] = df["batsman"].str.lower()
    df["bowler"] = df["bowler"].str.lower()

    row = df[(df["batsman"] == norm(batsman)) &
             (df["bowler"] == norm(bowler))]

    if len(row):
        return row.iloc[0]["strike_rate"], 25

    return 130, 25

def get_venue_avg(venue):
    df = main_df.copy()

    row = df[df["venue"].str.lower() == venue.lower()]

    if len(row):
        return row["venue_avg"].mean()

    return 160


def _safe_numeric(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def retrieve_similar_cases(
    score,
    wickets,
    over,
    balls,
    target_runs,
    predict_overs,
    is_chasing,
    striker,
    bowler,
    venue,
    limit=5
):
    df = main_df.copy()

    ball_number = int(over) * 6 + int(balls) + 1
    query_current_rr = score / max(over + balls / 6.0, 0.1)
    query_required_rr = target_runs / max(predict_overs, 1)

    df["ball_gap"] = (pd.to_numeric(df["ball_number"], errors="coerce") - ball_number).abs()
    df["score_gap"] = (pd.to_numeric(df["current_score"], errors="coerce") - score).abs()
    df["wicket_gap"] = (pd.to_numeric(df["wickets"], errors="coerce") - wickets).abs()
    df["rr_gap"] = (pd.to_numeric(df["current_rr"], errors="coerce") - query_current_rr).abs()
    df["req_gap"] = (pd.to_numeric(df["required_rr"], errors="coerce") - query_required_rr).abs()
    df["chasing_gap"] = (pd.to_numeric(df["is_chasing"], errors="coerce").fillna(0).astype(int) - int(is_chasing)).abs()

    df["similarity_score"] = (
        df["ball_gap"] * 0.55 +
        df["score_gap"] * 0.20 +
        df["wicket_gap"] * 8.0 +
        df["rr_gap"] * 7.5 +
        df["req_gap"] * 6.5 +
        df["chasing_gap"] * 20.0
    )

    striker_norm = norm(striker)
    bowler_norm = norm(bowler)
    venue_norm = norm(venue)

    df["batsman_norm"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler_norm"] = df["bowler"].astype(str).str.lower().str.strip()
    df["venue_norm"] = df["venue"].astype(str).str.lower().str.strip()

    df.loc[df["batsman_norm"] == striker_norm, "similarity_score"] -= 6.0
    df.loc[df["bowler_norm"] == bowler_norm, "similarity_score"] -= 4.0
    df.loc[df["venue_norm"] == venue_norm, "similarity_score"] -= 5.0

    top = df.sort_values("similarity_score").head(limit).copy()

    if top.empty:
        return {
            "empirical_probability": None,
            "avg_runs_in_window": None,
            "examples": []
        }

    balls_window = int(max(predict_overs, 1) * 6)
    if balls_window == 36 and "runs_next_36" in top.columns:
        top["window_runs"] = pd.to_numeric(top["runs_next_36"], errors="coerce")
    else:
        run_rate = pd.to_numeric(top["current_rr"], errors="coerce").fillna(query_current_rr)
        top["window_runs"] = run_rate * float(predict_overs)

    top["window_success"] = top["window_runs"] >= float(target_runs)

    empirical_probability = float(top["window_success"].mean()) if len(top) else None
    avg_runs_in_window = float(top["window_runs"].mean()) if len(top) else None

    examples = []
    for _, row in top.iterrows():
        examples.append({
            "date": str(row.get("date", "")),
            "venue": str(row.get("venue", "")),
            "batsman": str(row.get("batsman", "")),
            "bowler": str(row.get("bowler", "")),
            "current_score": int(_safe_numeric(row.get("current_score"), 0)),
            "wickets": int(_safe_numeric(row.get("wickets"), 0)),
            "ball_number": int(_safe_numeric(row.get("ball_number"), 0)),
            "window_runs": round(_safe_numeric(row.get("window_runs"), 0), 1),
            "success": bool(row.get("window_success", False))
        })

    return {
        "empirical_probability": empirical_probability,
        "avg_runs_in_window": avg_runs_in_window,
        "examples": examples
    }


def build_historical_evidence(striker, bowler, venue, target_runs, predict_overs, is_chasing):
    df = main_df.copy()

    striker_norm = norm(striker)
    bowler_norm = norm(bowler)
    venue_norm = norm(venue)
    balls_window = int(max(predict_overs, 1) * 6)

    df["batsman_norm"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler_norm"] = df["bowler"].astype(str).str.lower().str.strip()
    df["venue_norm"] = df["venue"].astype(str).str.lower().str.strip()
    df["is_chasing_num"] = pd.to_numeric(df["is_chasing"], errors="coerce").fillna(0).astype(int)

    if balls_window == 36 and "runs_next_36" in df.columns:
        df["window_runs"] = pd.to_numeric(df["runs_next_36"], errors="coerce")
    else:
        df["window_runs"] = pd.to_numeric(df["current_rr"], errors="coerce").fillna(0) * float(predict_overs)

    def summarize(subset, label):
        subset = subset.dropna(subset=["window_runs"])
        total = len(subset)
        if total == 0:
            return None
        successes = int((subset["window_runs"] >= float(target_runs)).sum())
        avg_runs = float(subset["window_runs"].mean())
        return {
            "label": label,
            "total": total,
            "successes": successes,
            "avg_runs": avg_runs,
            "summary": (
                f"{successes} out of {total} {label} reached {target_runs}+ runs in the next "
                f"{predict_overs} overs, with an average of {avg_runs:.1f}."
            )
        }

    filters = [
        (
            (df["batsman_norm"] == striker_norm) &
            (df["venue_norm"] == venue_norm) &
            (df["is_chasing_num"] == int(is_chasing)),
            f"historical {striker} at {venue} situations"
        ),
        (
            (df["batsman_norm"] == striker_norm) &
            (df["bowler_norm"] == bowler_norm) &
            (df["is_chasing_num"] == int(is_chasing)),
            f"historical {striker} vs {bowler} situations"
        ),
        (
            (df["venue_norm"] == venue_norm) &
            (df["is_chasing_num"] == int(is_chasing)),
            f"historical {venue} situations"
        ),
    ]

    evidence = []
    seen_labels = set()
    for mask, label in filters:
        item = summarize(df[mask].copy(), label)
        if item and item["label"] not in seen_labels:
            evidence.append(item)
            seen_labels.add(item["label"])

    return evidence
