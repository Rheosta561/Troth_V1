from pathlib import Path

from flask import Flask, request, jsonify
import joblib
import numpy as np
import torch
import torch.nn as nn

from commentary import compute_momentum
from live_stats import fetch_live_match_context
from utils import (
    build_historical_evidence,
    get_player_form,
    get_vs,
    get_venue_avg,
    retrieve_similar_scenarios,
)

try:
    from agent import agent
except Exception:
    agent = None

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


scaler = joblib.load(BASE_DIR / "scaler.pkl")
player_map = joblib.load(BASE_DIR / "player_map.pkl")
bowler_map = joblib.load(BASE_DIR / "bowler_map.pkl")


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.player_emb = nn.Embedding(len(player_map), 16)
        self.bowler_emb = nn.Embedding(len(bowler_map), 16)

        self.net = nn.Sequential(
            nn.Linear(12 + 32, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, p, b):
        p_emb = self.player_emb(p)
        b_emb = self.bowler_emb(b)
        x = torch.cat([x, p_emb, b_emb], dim=1)
        return self.net(x)


model = Model()
model.load_state_dict(torch.load(BASE_DIR / "model.pth", map_location="cpu"))
model.eval()


def safe(data, key, default):
    return data.get(key, default)


def clamp_probability(value):
    return float(np.clip(value, 0.01, 0.99))


def sanitize_rate(value, fallback):
    try:
        if value is None:
            return None if fallback is None else float(fallback)
        numeric = float(value)
    except Exception:
        return None if fallback is None else float(fallback)

    if numeric <= 0 or numeric > 300:
        return None if fallback is None else float(fallback)
    return numeric


def display_rate(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def compute_probability(predicted_runs, target_runs, similarity_result, pressure_index, is_powerplay):
    confidence_band = max(target_runs * 0.2, 8)
    run_gap_score = 0.5 + ((predicted_runs - target_runs) / (2 * confidence_band))
    run_gap_score = float(np.clip(run_gap_score, 0.05, 0.95))

    # Tighten probability when the ask is much steeper than the current pace,
    # but allow a slight boost in powerplay when wickets are intact.
    pressure_adjustment = 1.0 - min(max(pressure_index - 1.0, 0.0) * 0.12, 0.22)
    powerplay_boost = 1.04 if is_powerplay else 1.0
    run_gap_score = float(np.clip(run_gap_score * pressure_adjustment * powerplay_boost, 0.03, 0.97))

    success_rate = similarity_result.get("success_rate")
    if success_rate is None:
        return clamp_probability(run_gap_score)

    sample_size = len(similarity_result.get("examples", []))
    exact_match_bonus = 0.08 if similarity_result.get("exact_batsman_bowler_count", 0) > 0 else 0.0
    history_weight = min(0.18 + sample_size * 0.07 + exact_match_bonus, 0.58)
    blended = (1 - history_weight) * run_gap_score + history_weight * float(success_rate)
    return clamp_probability(blended)


def probability_to_prediction(prob):
    if prob >= 0.78:
        return "STRONG YES"
    if prob >= 0.6:
        return "YES"
    if prob >= 0.45:
        return "RISKY"
    return "NO"


def build_historical_evidence_text(evidence_items):
    if evidence_items:
        return "\n".join(evidence_items)
    return "No direct batsman, bowler, or venue evidence was available in the current dataset."


def build_deterministic_analysis(
    prediction,
    prob,
    score,
    wickets,
    over,
    balls,
    batting_team,
    bowling_team,
    striker,
    non_striker,
    bowler,
    target_runs,
    target_end_over,
    runs_needed,
    window_balls,
    current_rr,
    required_rr_window,
    pressure_index,
    is_powerplay,
    striker_form_display,
    non_striker_form_display,
    vs_sr_display,
    model_runs,
    historical_runs,
    predicted_runs,
    similarity_summary,
    historical_evidence,
):
    outcome_line = f"Model verdict: {prediction} ({prob * 100:.1f}%)."
    context_line = (
        f"{batting_team} are {score}/{wickets} after {over}.{balls}, facing {bowling_team}. "
        f"They need {runs_needed} more runs in {window_balls} balls to reach {target_runs} by the end of over {target_end_over}."
    )
    rate_line = (
        f"Current RR is {current_rr:.2f} and required RR in the window is {required_rr_window:.2f}. "
        f"Pressure index is {pressure_index:.2f}, with powerplay {'active' if is_powerplay else 'off'}."
    )
    player_line = (
        f"{striker} form SR: {display_rate(striker_form_display)}. "
        f"{non_striker} form SR: {display_rate(non_striker_form_display)}. "
        f"{striker} vs {bowler} SR: {display_rate(vs_sr_display)}."
    )
    forecast_line = (
        f"Model forecast is {model_runs:.1f} runs for the window. "
        f"Historical similarity forecast is {historical_runs:.1f} runs." if historical_runs is not None
        else f"Model forecast is {model_runs:.1f} runs for the window."
    )
    blended_line = f"Blended projection is {predicted_runs:.1f} runs in this scoring window."
    evidence_block = f"Historical evidence: {historical_evidence}"
    similarity_block = f"Closest match-state summary: {similarity_summary}"
    final_line = f"Final answer: {prediction}"

    return "\n\n".join([
        outcome_line,
        context_line,
        rate_line,
        player_line,
        forecast_line,
        blended_line,
        evidence_block,
        similarity_block,
        final_line,
    ])


def build_similarity_summary(similarity_result, target_runs, predict_overs):
    examples = similarity_result.get("examples", [])
    success_rate = similarity_result.get("success_rate")
    predicted_runs = similarity_result.get("predicted_runs")

    if not examples:
        return "No close match-state examples were found in the dynamic dataset."

    lines = []
    if success_rate is not None:
        successes = int(round(success_rate * len(examples)))
        lines.append(
            f"{successes} out of {len(examples)} closest match states reached {target_runs}+ runs in the next {predict_overs} overs."
        )
    if predicted_runs is not None:
        lines.append(f"Average projected runs from those similar states: {predicted_runs:.1f}.")

    for ex in examples[:3]:
        verdict = "hit" if ex["success"] else "missed"
        lines.append(
            f"{ex['batsman']} vs {ex['bowler']} at {ex['venue']} from {ex['score']}/{ex['wickets']} after "
            f"{ex['over']}.{ex['ball']} {verdict} with {ex['window_runs']:.1f} runs in that window."
        )

    return "\n".join(lines)


def build_search_queries(striker, non_striker, bowler, venue):
    return [
        f"{striker} batting record {venue} IPL",
        f"{striker} vs {bowler} IPL",
        f"{non_striker} batting record {venue} IPL",
        f"{bowler} bowling record {venue} IPL",
    ]


def compute_context_adjustment(current_rr, target_runs, window_balls, striker_form_sr, non_striker_form_sr, vs_sr, pressure_index, is_powerplay, wickets):
    required_rr = target_runs / max(float(window_balls) / 6.0, 0.1)
    chase_pressure = required_rr - current_rr
    striker_form_sr = striker_form_sr if striker_form_sr is not None else 115.0
    non_striker_form_sr = non_striker_form_sr if non_striker_form_sr is not None else striker_form_sr
    vs_sr = vs_sr if vs_sr is not None else striker_form_sr

    batting_support = (
        0.45 * (striker_form_sr / 100.0) +
        0.25 * (non_striker_form_sr / 100.0) +
        0.30 * (vs_sr / 100.0)
    )
    batting_support = np.clip(batting_support, 0.7, 1.8)

    pressure_factor = np.clip(1.0 - (chase_pressure / max(required_rr, 1.0)) * 0.35, 0.72, 1.18)
    explicit_pressure_factor = np.clip(1.06 - max(pressure_index - 1.0, 0.0) * 0.16, 0.7, 1.08)
    powerplay_factor = 1.05 if is_powerplay and wickets <= 2 else 1.0
    wicket_factor = np.clip(1.0 - max(wickets - 2, 0) * 0.05, 0.8, 1.0)
    return float(batting_support * pressure_factor * explicit_pressure_factor * powerplay_factor * wicket_factor)


@app.route("/predict-live", methods=["POST"])
def predict():
    data = request.json or {}
    request_source = safe(data, "source", "live")

    score = safe(data, "score", 0)
    wickets = safe(data, "wickets", 0)
    over = safe(data, "over", 0)
    balls = safe(data, "balls", 0)
    target_runs = safe(data, "target_runs", 0)
    target_end_over = max(safe(data, "overs", 6), 0)
    striker = safe(data, "striker", "Unknown")
    non_striker = safe(data, "non_striker", "Unknown")
    bowler = safe(data, "bowler", "Unknown")
    batting_team = safe(data, "batting_team", "Mumbai Indians")
    bowling_team = safe(data, "bowling_team", "Kolkata Knight Riders")
    venue = safe(data, "venue", "Unknown")
    target_total = safe(data, "target_total", 0)

    live_context = fetch_live_match_context(batting_team, bowling_team)
    require_live_validation = safe(data, "live", True) and request_source != "manual"
    if require_live_validation and not live_context.get("live", False):
        return jsonify({
            "status": "no_live_match",
            "message": f"No live match found for {batting_team} vs {bowling_team} on ESPN or Cricbuzz."
        })
    balls_done = over * 6 + balls
    runs_needed = max(target_runs - score, 0)
    target_window_balls = int(target_end_over * 6)
    window_balls = max(target_window_balls - balls_done, 0)
    window_overs = max(window_balls / 6.0, 0)

    balls_left_innings = max(120 - balls_done, 0)
    current_rr = score / max(over + balls / 6, 0.1)

    runs_last_6 = max(int(round(current_rr)), 0)
    runs_last_12 = max(int(round(current_rr * 2)), runs_last_6)

    try:
        live_runs_6, live_runs_12, detected = compute_momentum()
        runs_last_6 = live_runs_6
        runs_last_12 = live_runs_12
        if detected != "unknown":
            striker = detected
    except Exception:
        pass

    momentum = runs_last_6 + runs_last_12
    acceleration = runs_last_6 / (runs_last_12 + 1)
    wickets_left = 10 - wickets
    phase = 0 if over < 6 else (1 if over < 15 else 2)
    is_powerplay = over < 6
    run_rate_trend = runs_last_6 - runs_last_12 / 2
    wicket_pressure = wickets / (balls_done + 1)
    required_rr_window = runs_needed / max(float(window_balls) / 6.0, 0.1) if window_balls > 0 else 0.0
    pressure_index = required_rr_window / max(current_rr, 0.1) if runs_needed > 0 else 0.0

    form_sr = get_player_form(striker)
    non_striker_form_sr = get_player_form(non_striker)
    vs_sr, _ = get_vs(striker, bowler)
    venue_avg = get_venue_avg(venue)
    form_sr = sanitize_rate(form_sr, None)
    non_striker_form_sr = sanitize_rate(non_striker_form_sr, None)
    vs_sr = sanitize_rate(vs_sr, None)
    striker_form_display = form_sr
    non_striker_form_display = non_striker_form_sr
    vs_sr_display = vs_sr

    features = np.array([[
        score, wickets, wickets_left,
        balls_left_innings, phase,
        runs_last_6, runs_last_12,
        momentum, acceleration, run_rate_trend,
        current_rr, wicket_pressure
    ]])

    features = scaler.transform(features)
    tensor = torch.tensor(features, dtype=torch.float32)

    batsman_id = player_map.get(striker, 0)
    bowler_id = bowler_map.get(bowler, 0)
    batsman_tensor = torch.tensor([batsman_id], dtype=torch.long)
    bowler_tensor = torch.tensor([bowler_id], dtype=torch.long)

    with torch.no_grad():
        model_runs = float(model(tensor, batsman_tensor, bowler_tensor).item() * 100)

    similarity_result = retrieve_similar_scenarios(
        score=score,
        wickets=wickets,
        over=over,
        balls=balls,
        target_runs=runs_needed,
        predict_overs=max(window_overs, 0.1),
        striker=striker,
        bowler=bowler,
        venue=venue,
    )
    historical_runs = similarity_result.get("predicted_runs")

    if historical_runs is not None:
        predicted_runs = 0.6 * model_runs + 0.4 * historical_runs
    else:
        predicted_runs = model_runs

    context_multiplier = compute_context_adjustment(
        current_rr=current_rr,
        target_runs=runs_needed,
        window_balls=max(window_balls, 1),
        striker_form_sr=form_sr,
        non_striker_form_sr=non_striker_form_sr,
        vs_sr=vs_sr,
        pressure_index=pressure_index,
        is_powerplay=is_powerplay,
        wickets=wickets
    )
    predicted_runs *= context_multiplier

    if runs_needed <= 0:
        prob = 0.99
    else:
        prob = compute_probability(predicted_runs, runs_needed, similarity_result, pressure_index, is_powerplay)

    prediction = probability_to_prediction(prob)

    evidence_items = build_historical_evidence(
        striker=striker,
        bowler=bowler,
        venue=venue,
        target_runs=runs_needed,
        predict_overs=max(window_overs, 0.1),
    )
    historical_evidence = build_historical_evidence_text(evidence_items)
    similarity_summary = build_similarity_summary(similarity_result, runs_needed, max(window_overs, 0.1))

    analysis = build_deterministic_analysis(
        prediction=prediction,
        prob=prob,
        score=score,
        wickets=wickets,
        over=over,
        balls=balls,
        batting_team=batting_team,
        bowling_team=bowling_team,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        target_runs=target_runs,
        target_end_over=target_end_over,
        runs_needed=runs_needed,
        window_balls=window_balls,
        current_rr=current_rr,
        required_rr_window=required_rr_window,
        pressure_index=pressure_index,
        is_powerplay=is_powerplay,
        striker_form_display=striker_form_display,
        non_striker_form_display=non_striker_form_display,
        vs_sr_display=vs_sr_display,
        model_runs=model_runs,
        historical_runs=historical_runs,
        predicted_runs=predicted_runs,
        similarity_summary=similarity_summary,
        historical_evidence=historical_evidence,
    )

    return jsonify({
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "live_sources": {
            "espn": bool(live_context.get("espn")),
            "cricbuzz": bool(live_context.get("cricbuzz")),
        },
        "runs_needed": runs_needed,
        "window_balls": window_balls,
        "window_overs": round(window_overs, 2),
        "target_end_over": target_end_over,
        "is_powerplay": is_powerplay,
        "pressure_index": round(pressure_index, 3),
        "predicted_runs": round(predicted_runs, 2),
        "model_predicted_runs": round(model_runs, 2),
        "historical_predicted_runs": round(historical_runs, 2) if historical_runs is not None else None,
        "prediction": prediction,
        "probability": round(prob, 3),
        "historical_evidence": historical_evidence,
        "analysis": analysis,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
