from pathlib import Path
from difflib import get_close_matches
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def _load_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return pd.DataFrame(), False
    return pd.read_csv(path), True


player_stats, HAS_PLAYER_STATS = _load_csv("srh_kkr_player_stats_enhanced.csv")
vs_stats, HAS_VS_STATS = _load_csv("srh_kkr_vs_stats_enhanced.csv")


def _build_srh_kkr_main_df():
    kkr_df, has_kkr = _load_csv("mi_kkr_dynamic_dataset.csv")
    srh_source, has_srh = _load_csv("rcb_srh_final_dataset.csv")

    frames = []

    if has_kkr and not kkr_df.empty:
        frames.append(kkr_df.copy())

    if has_srh and not srh_source.empty:
        srh_df = srh_source.copy()
        srh_df["batting_team_norm"] = srh_df["batting_team"].astype(str).str.lower().str.strip()
        srh_df = srh_df[srh_df["batting_team_norm"].str.contains("sunrisers hyderabad", na=False)].copy()

        if not srh_df.empty:
            srh_df["future_runs_6"] = pd.to_numeric(srh_df.get("runs_next_36"), errors="coerce")
            srh_df["future_runs_12"] = np.nan
            srh_df["future_runs_18"] = np.nan
            frames.append(
                srh_df[[
                    "venue", "batsman", "bowler", "over", "ball", "current_score",
                    "wickets", "runs_last_6", "runs_last_12", "current_rr",
                    "future_runs_6", "future_runs_12", "future_runs_18"
                ]].copy()
            )

    if not frames:
        return pd.DataFrame(), False

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, not combined.empty


main_df, HAS_MAIN_DF = _build_srh_kkr_main_df()


def norm(x):
    return str(x).lower().strip()


def normalize_person_key(name):
    value = norm(name)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("deshpandey", "deshpande")
    value = value.replace("rickleton", "rickelton")
    value = value.replace("mahatre", "mhatre")
    return value


VENUE_ALIASES = {
    "arun jaitley delhi": "arun jaitley",
    "arun jaitley stadium delhi": "arun jaitley",
    "arun jaitley stadium": "arun jaitley",
    "firoz shah kotla": "arun jaitley",
    "firoz shah kotla delhi": "arun jaitley",
    "kotla": "arun jaitley",
    "wankhede mumbai": "wankhede",
    "wankhede stadium mumbai": "wankhede",
    "wankhede stadium": "wankhede",
    "ma chidambaram chepauk chennai": "chepauk",
    "ma chidambaram stadium chepauk chennai": "chepauk",
    "chepauk": "chepauk",
    "eden gardens kolkata": "eden gardens",
    "m chinnaswamy stadium bengaluru": "chinnaswamy",
    "m chinnaswamy stadium bangalore": "chinnaswamy",
    "chinnaswamy": "chinnaswamy",
    "sawai mansingh stadium jaipur": "sawai mansingh",
    "rajiv gandhi international stadium uppal hyderabad": "uppal",
    "narendra modi stadium ahmedabad": "narendra modi",
    "motera ahmedabad": "narendra modi",
    "motera": "narendra modi",
    "maharaja yadavindra singh international cricket stadium": "mullanpur",
    "maharaja yadavindra singh international cricket stadium mullanpur": "mullanpur",
    "new chandigarh mullanpur": "mullanpur",
    "mullanpur": "mullanpur",
    "ekana cricket stadium": "ekana",
    "ekana cricket stadium lucknow": "ekana",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "ekana",
    "brsabv ekana cricket stadium": "ekana",
    "lucknow": "ekana",
    "rajiv gandhi international stadium": "uppal",
    "rajiv gandhi international stadium hyderabad": "uppal",
    "rajiv gandhi international stadium uppal": "uppal",
    "uppal hyderabad": "uppal",
}

PLAYER_ALIASES = {
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "srh": "Sunrisers Hyderabad",
    "kolkata knight riders": "Kolkata Knight Riders",
    "kkr": "Kolkata Knight Riders",
    "ishan kishan": "Ishan Kishan",
    "kishan": "Ishan Kishan",
    "heinrich klaasen": "H Klaasen",
    "klaasen": "H Klaasen",
    "travis head": "TM Head",
    "head": "TM Head",
    "abhishek sharma": "Abhishek Sharma",
    "nitish kumar reddy": "Nithish Kumar Reddy",
    "nithish kumar reddy": "Nithish Kumar Reddy",
    "nitish reddy": "Nithish Kumar Reddy",
    "aiden markram": "AK Markram",
    "markram": "AK Markram",
    "abdul samad": "Abdul Samad",
    "kamindu mendis": "Kamindu Mendis",
    "harshal patel": "Harshal Patel",
    "pat cummins": "PJ Cummins",
    "cummins": "PJ Cummins",
    "jaydev unadkat": "JD Unadkat",
    "unadkat": "JD Unadkat",
    "shivam mavi": "Shivam Mavi",
    "umran malik": "Umran Malik",
    "zeeshan ansari": "AS Roy",
    "ajinkya rahane": "AM Rahane",
    "rahane": "AM Rahane",
    "rinku singh": "RK Singh",
    "rinku": "RK Singh",
    "angkrish raghuvanshi": "A Raghuvanshi",
    "angkrish raghuwanshi": "A Raghuvanshi",
    "raghuvanshi": "A Raghuvanshi",
    "manish pandey": "MK Pandey",
    "pandey": "MK Pandey",
    "finn allen": "Finn Allen",
    "rahul tripathi": "RA Tripathi",
    "tripathi": "RA Tripathi",
    "tim seifert": "TL Seifert",
    "rovman powell": "R Powell",
    "powell": "R Powell",
    "anukul roy": "AS Roy",
    "cameron green": "C Green",
    "rachin ravindra": "R Ravindra",
    "ramandeep singh": "Ramandeep Singh",
    "sunil narine": "SP Narine",
    "narine": "SP Narine",
    "vaibhav arora": "VG Arora",
    "arora": "VG Arora",
    "kartik tyagi": "Kartik Tyagi",
    "varun chakaravarthy": "CV Varun",
    "varun chakravarthy": "CV Varun",
    "varun": "CV Varun",
    "umran malik": "Umran Malik",
    "harshit rana": "Harshit Rana",
    "matheesha pathirana": "Matheesha Pathirana",
}


def normalize_venue_name(venue):
    value = norm(venue)
    value = value.replace(",", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    tokens = [token for token in value.split() if token not in {"stadium", "cricket", "ground"}]
    cleaned = " ".join(tokens)
    cleaned = cleaned.replace("jaitely", "jaitley")
    return VENUE_ALIASES.get(cleaned, cleaned)


def build_name_index(names):
    index = {}
    for raw_name in names:
        raw_name = str(raw_name).strip()
        if not raw_name:
            continue
        key = normalize_person_key(raw_name)
        index.setdefault(key, raw_name)

        parts = key.split()
        if parts:
            surname = parts[-1]
            index.setdefault(surname, raw_name)
            index.setdefault(" ".join(parts[-2:]), raw_name)
            if len(parts) >= 2:
                first_initial = parts[0][0]
                index.setdefault(f"{first_initial} {surname}", raw_name)
                index.setdefault(f"{first_initial}{surname}", raw_name)

            compact = "".join(parts)
            index.setdefault(compact, raw_name)

    return index


def score_name_match(query_key, candidate_key):
    query_parts = query_key.split()
    candidate_parts = candidate_key.split()
    if not query_parts or not candidate_parts:
        return 0.0

    score = 0.0
    if query_key == candidate_key:
        score += 1.0

    query_surname = query_parts[-1]
    candidate_surname = candidate_parts[-1]
    if query_surname == candidate_surname:
        score += 0.75
    elif query_surname in candidate_surname or candidate_surname in query_surname:
        score += 0.45

    if query_parts[0][0] == candidate_parts[0][0]:
        score += 0.2

    overlap = len(set(query_parts) & set(candidate_parts))
    score += 0.18 * overlap

    compact_query = "".join(query_parts)
    compact_candidate = "".join(candidate_parts)
    if compact_query == compact_candidate:
        score += 0.5

    close = get_close_matches(query_key, [candidate_key], n=1, cutoff=0.7)
    if close:
        score += 0.25

    return score


PLAYER_NAME_INDEX = build_name_index(player_stats["player"].tolist()) if HAS_PLAYER_STATS and not player_stats.empty else {}
VS_BATSMAN_INDEX = build_name_index(vs_stats["batsman"].tolist()) if HAS_VS_STATS and not vs_stats.empty else {}
VS_BOWLER_INDEX = build_name_index(vs_stats["bowler"].tolist()) if HAS_VS_STATS and not vs_stats.empty else {}
MAIN_BATSMAN_INDEX = build_name_index(main_df["batsman"].tolist()) if HAS_MAIN_DF and not main_df.empty else {}
MAIN_BOWLER_INDEX = build_name_index(main_df["bowler"].tolist()) if HAS_MAIN_DF and not main_df.empty else {}


def resolve_name(name, index):
    key = normalize_person_key(name)
    if not key:
        return str(name)
    if key in PLAYER_ALIASES:
        return PLAYER_ALIASES[key]
    if key in index:
        return index[key]

    compact = "".join(key.split())
    if compact in index:
        return index[compact]

    parts = key.split()
    if len(parts) >= 2:
        initial_surname = f"{parts[0][0]} {parts[-1]}"
        if initial_surname in index:
            return index[initial_surname]
        initial_surname_compact = f"{parts[0][0]}{parts[-1]}"
        if initial_surname_compact in index:
            return index[initial_surname_compact]

    matches = get_close_matches(key, list(index.keys()), n=1, cutoff=0.82)
    if matches:
        return index[matches[0]]

    candidates = list(dict.fromkeys(index.values()))
    if candidates:
        best_name = str(name)
        best_score = 0.0
        for candidate in candidates:
            candidate_key = normalize_person_key(candidate)
            score = score_name_match(key, candidate_key)
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_score >= 0.7:
            return best_name
    return str(name)


def get_player_form(name):
    if not HAS_PLAYER_STATS or player_stats.empty:
        return None
    df = player_stats.copy()
    df["player"] = df["player"].astype(str).str.lower().str.strip()
    resolved_name = resolve_name(name, PLAYER_NAME_INDEX)

    row = df[df["player"] == norm(resolved_name)]
    return float(row.iloc[0]["strike_rate"]) if len(row) else None


def get_vs(batsman, bowler):
    if not HAS_VS_STATS or vs_stats.empty:
        return None, None
    df = vs_stats.copy()
    df["batsman"] = df["batsman"].astype(str).str.lower().str.strip()
    df["bowler"] = df["bowler"].astype(str).str.lower().str.strip()
    resolved_batsman = resolve_name(batsman, VS_BATSMAN_INDEX)
    resolved_bowler = resolve_name(bowler, VS_BOWLER_INDEX)

    row = df[
        (df["batsman"] == norm(resolved_batsman)) &
        (df["bowler"] == norm(resolved_bowler))
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
    striker_norm = norm(resolve_name(striker, MAIN_BATSMAN_INDEX))
    bowler_norm = norm(resolve_name(bowler, MAIN_BOWLER_INDEX))

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
    evidence = []
    striker_resolved = resolve_name(striker, PLAYER_NAME_INDEX or MAIN_BATSMAN_INDEX)
    bowler_resolved = resolve_name(bowler, PLAYER_NAME_INDEX or MAIN_BOWLER_INDEX)
    venue_norm = normalize_venue_name(venue)

    if HAS_VS_STATS and not vs_stats.empty:
        vs_df = vs_stats.copy()
        vs_df["batsman_norm"] = vs_df["batsman"].astype(str).str.lower().str.strip()
        vs_df["bowler_norm"] = vs_df["bowler"].astype(str).str.lower().str.strip()
        exact = vs_df[
            (vs_df["batsman_norm"] == norm(resolve_name(striker, VS_BATSMAN_INDEX))) &
            (vs_df["bowler_norm"] == norm(resolve_name(bowler, VS_BOWLER_INDEX)))
        ]
        if not exact.empty:
            row = exact.iloc[0]
            evidence.append(
                f"{striker_resolved} vs {bowler_resolved}: {int(_to_float(row.get('runs'), 0))} runs from "
                f"{int(_to_float(row.get('balls'), 0))} balls at SR {float(row.get('strike_rate', 0)):.1f}, "
                f"with {int(_to_float(row.get('dismissals'), 0))} dismissals."
            )

    if HAS_PLAYER_STATS and not player_stats.empty:
        ps_df = player_stats.copy()
        ps_df["player_norm"] = ps_df["player"].astype(str).str.lower().str.strip()

        striker_row = ps_df[ps_df["player_norm"] == norm(striker_resolved)]
        if not striker_row.empty:
            row = striker_row.iloc[0]
            evidence.append(
                f"{striker_resolved} profile: overall SR {float(row.get('strike_rate', 0)):.1f}, "
                f"powerplay SR {float(row.get('pp_sr', 0)):.1f}, middle-overs SR {float(row.get('middle_sr', 0)):.1f}, "
                f"death-overs SR {float(row.get('death_sr', 0)):.1f}."
            )

        bowler_row = ps_df[ps_df["player_norm"] == norm(bowler_resolved)]
        if not bowler_row.empty and float(bowler_row.iloc[0].get("balls_bowled", 0) or 0) > 0:
            row = bowler_row.iloc[0]
            evidence.append(
                f"{bowler_resolved} bowling profile: economy {float(row.get('economy', 0)):.2f}, "
                f"bowling SR {float(row.get('bowling_sr', 0)):.1f}, dot-ball rate {float(row.get('dot_bowl_pct', 0)) * 100:.1f}%."
            )

    if HAS_MAIN_DF and not main_df.empty:
        df = main_df.copy()
        df["venue_norm"] = df["venue"].astype(str).apply(normalize_venue_name)
        df["batsman_norm"] = df["batsman"].astype(str).str.lower().str.strip()
        df["bowler_norm"] = df["bowler"].astype(str).str.lower().str.strip()

        future_col, supported_window = get_future_runs_column(predict_overs)
        df["window_runs"] = pd.to_numeric(df[future_col], errors="coerce")
        if int(max(round(float(predict_overs)), 1)) != supported_window:
            df["window_runs"] = df["window_runs"] * (float(predict_overs) / supported_window)

        striker_norm = norm(resolve_name(striker, MAIN_BATSMAN_INDEX))
        bowler_norm = norm(resolve_name(bowler, MAIN_BOWLER_INDEX))

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

        for mask, label in [
            ((df["batsman_norm"] == striker_norm) & (df["bowler_norm"] == bowler_norm),
             f"{striker_resolved} vs {bowler_resolved} match states"),
            ((df["batsman_norm"] == striker_norm) & (df["venue_norm"] == venue_norm),
             f"{striker_resolved} at {venue} situations"),
            ((df["venue_norm"] == venue_norm),
             f"{venue} situations"),
        ]:
            summary = summarize(mask, label)
            if summary:
                evidence.append(summary)

    return evidence[:4]
