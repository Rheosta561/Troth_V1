from pathlib import Path

from flask import Flask, request, jsonify
import torch, joblib, numpy as np
import torch.nn as nn

from commentary import compute_momentum
from utils import get_player_form, get_vs, get_venue_avg, retrieve_similar_cases, build_historical_evidence

try:
    from agent import agent
except Exception:
    agent = None

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# =========================
# LOAD SCALER
# =========================
scaler = joblib.load(BASE_DIR / "scaler.pkl")

# =========================
# MODEL (19 FEATURES)
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(19, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = Model()
model.load_state_dict(torch.load(BASE_DIR / "model.pth", map_location="cpu"))
model.eval()

# =========================
# SAFE GETTER
# =========================
def safe(data, key, default):
    return data.get(key, default)


def build_fallback_analysis(score, wickets, over, balls, target_runs, predict_overs, striker, bowler, venue, prob):
    over_text = f"{over}.{balls}"
    verdict = "YES" if prob > 0.45 else "NO"
    return (
        f"Score is {score}/{wickets} after {over_text} overs at {venue}. "
        f"{striker} vs {bowler} projects a {prob * 100:.1f}% chance of scoring "
        f"{target_runs} runs in the next {predict_overs} overs. Final call: {verdict}."
    )


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def calibrate_probability(base_prob, target_runs, predict_overs, current_rr, wickets, venue_avg, form, vs_sr):
    safe_overs = max(float(predict_overs), 1.0)
    ask_rr = float(target_runs) / safe_overs

    venue_rr = float(venue_avg) / 20.0
    batter_rr = max(float(vs_sr) / 100.0, 0.5)
    form_boost = max(float(form) / 35.0, 0.5)
    wicket_penalty = 1.0 + max(float(wickets) - 2.0, 0.0) * 0.12

    expected_rr = max(current_rr, 0.1) * 0.45 + venue_rr * 0.35 + batter_rr * form_boost * 0.20
    expected_rr = max(expected_rr / wicket_penalty, 0.1)

    difficulty = ask_rr / expected_rr

    clipped_base = float(np.clip(base_prob, 0.05, 0.95))
    logit = np.log(clipped_base / (1.0 - clipped_base))

    adjusted_logit = logit - 1.35 * (difficulty - 1.0)
    adjusted_prob = 1.0 / (1.0 + np.exp(-adjusted_logit))

    return float(np.clip(adjusted_prob, 0.01, 0.99))


def blend_with_historical_probability(model_prob, retrieval_result, predict_overs):
    empirical = retrieval_result.get("empirical_probability")
    if empirical is None:
        return model_prob

    sample_size = len(retrieval_result.get("examples", []))
    if sample_size <= 0:
        return model_prob

    historical_weight = 0.45 if int(predict_overs) == 6 else 0.30
    historical_weight = min(historical_weight, 0.15 + sample_size * 0.06)

    blended = (1 - historical_weight) * model_prob + historical_weight * float(empirical)
    return float(np.clip(blended, 0.01, 0.99))


def build_retrieval_summary(retrieval_result, target_runs, predict_overs):
    examples = retrieval_result.get("examples", [])
    empirical = retrieval_result.get("empirical_probability")
    avg_runs = retrieval_result.get("avg_runs_in_window")

    if not examples:
        return "No close historical examples were found in the dataset."

    lines = []
    if empirical is not None:
        lines.append(
            f"Historical retrieval: {int(round(empirical * 100))}% of the {len(examples)} closest situations scored {target_runs}+ runs in the next {predict_overs} overs."
        )
    if avg_runs is not None:
        lines.append(
            f"Average runs from those similar windows: {avg_runs:.1f}."
        )

    for ex in examples[:3]:
        over_text = f"{(max(ex['ball_number'], 1) - 1) // 6}.{(max(ex['ball_number'], 1) - 1) % 6}"
        verdict = "hit" if ex["success"] else "missed"
        lines.append(
            f"{ex['date']} at {ex['venue']}: {ex['batsman']} vs {ex['bowler']} from {ex['current_score']}/{ex['wickets']} after {over_text} overs {verdict} with {ex['window_runs']:.1f} runs in that window."
        )

    return "\n".join(lines)


def format_historical_evidence(evidence_items):
    if not evidence_items:
        return "No direct player or venue evidence was found in the current dataset."
    return "\n".join(item["summary"] for item in evidence_items)


def build_agent_prompt(
    score,
    wickets,
    over,
    balls,
    striker,
    non_striker,
    bowler,
    venue,
    target_total,
    remaining_runs,
    innings_overs_left,
    predict_overs,
    context_label,
    current_rr_actual,
    required_rr_actual,
    target_runs,
    retrieval_summary,
    historical_evidence,
    prediction
):
    if context_label == "match chase context":
        situation_block = f"""
Match Target: {target_total}
Runs Remaining In Chase: {remaining_runs}
Innings Overs Left: {innings_overs_left:.2f}
Chase Required RR: {required_rr_actual:.2f}
"""
        framing = "This is a chase scenario."
    else:
        situation_block = f"""
Innings Overs Left: {innings_overs_left:.2f}
Bet Window Overs: {predict_overs}
Bet Target In Window: {target_runs} runs
Bet Required RR: {required_rr_actual:.2f}
"""
        framing = "This is not a chase scenario. Judge only whether the batting side can score the requested runs in the next overs window."

    return f"""
Match Situation:
Score: {score}/{wickets}
Over: {over}.{balls}

Striker: {striker}
Non-Striker: {non_striker}
Bowler: {bowler}
Venue: {venue}

{situation_block}
Current RR: {current_rr_actual:.2f}
Projection Context: {context_label}
{framing}

Bet:
Can team score {target_runs} runs in next {predict_overs} overs?

Historical retrieval from dataset:
{retrieval_summary}

Historical evidence:
{historical_evidence}

Give:
- tactical reasoning
- key factors
- final YES/NO aligned with the model verdict below
- do not describe this as a chase unless the context says match chase

Model verdict to honor: {prediction}
"""


def derive_projection_context(is_chasing, score, target_total, total_overs, over, balls, target_runs, predict_overs, current_rr):
    remaining_runs = max(target_total - score, 0)
    remaining_overs = max(total_overs - (over + balls / 6), 0.1)
    scenario_rr = max(float(target_runs) / max(float(predict_overs), 1.0), 0.1)

    if is_chasing:
        required_rr = remaining_runs / remaining_overs
        context_mode = "match_target"
    else:
        required_rr = scenario_rr
        remaining_runs = target_runs
        remaining_overs = predict_overs
        context_mode = "bet_target"

    pressure = required_rr / max(current_rr, 0.1)

    return required_rr, pressure, remaining_runs, remaining_overs, context_mode

# =========================
# API
# =========================
@app.route("/predict-live", methods=["POST"])
def predict():

    data = request.json or {}

    if not data.get("live", False):
        return jsonify({"status": "no_live_match"})

    # =========================
    # INPUTS
    # =========================
    score = safe(data, "score", 0)
    wickets = safe(data, "wickets", 0)

    over = safe(data, "over", 0)
    balls = safe(data, "balls", 0)

    target_total = safe(data, "target_total", 0)
    total_overs = safe(data, "total_overs", 20)

    target_runs = safe(data, "target_runs", 50)
    predict_overs = safe(data, "overs", 6)
    is_chasing = parse_bool(safe(data, "is_chasing", False))

    striker = safe(data, "striker", "unknown")
    non_striker = safe(data, "non_striker", "unknown")
    bowler = safe(data, "bowler", "unknown")
    venue = safe(data, "venue", "unknown")

    if is_chasing and target_total <= 0:
        return jsonify({
            "status": "invalid_input",
            "message": "Target total is required when the team is chasing."
        }), 400

    # =========================
    # CORE MATCH LOGIC
    # =========================
    balls_bowled = over * 6 + balls
    balls_left = predict_overs * 6

    current_rr = score / (over + balls/6 + 1e-5)

    required_rr, pressure, remaining_runs, remaining_overs, context_mode = derive_projection_context(
        is_chasing=is_chasing,
        score=score,
        target_total=target_total,
        total_overs=total_overs,
        over=over,
        balls=balls,
        target_runs=target_runs,
        predict_overs=predict_overs,
        current_rr=current_rr
    )

    # =========================
    # CLEAN + LOG
    # =========================
    required_rr = np.clip(required_rr, 0, 20)
    current_rr = np.clip(current_rr, 0, 20)
    pressure = np.clip(pressure, 0, 5)

    required_rr = np.log1p(required_rr)
    current_rr = np.log1p(current_rr)
    pressure = np.log1p(pressure)

    # =========================
    # MOMENTUM (REAL)
    # =========================
    runs_last_6 = max(int(round(current_rr)), 0)
    runs_last_12 = max(int(round(current_rr * 2)), runs_last_6)

    try:
        live_runs_6, live_runs_12, detected_striker = compute_momentum()

        runs_last_6 = live_runs_6
        runs_last_12 = live_runs_12

        if detected_striker != "unknown":
            striker = detected_striker

    except Exception:
        pass

    momentum = runs_last_6 + runs_last_12
    acceleration = runs_last_6 / (runs_last_12 + 1)

    wickets_left = 10 - wickets

    # =========================
    # PLAYER FEATURES
    # =========================
    form = get_player_form(striker)
    vs_sr, vs_avg = get_vs(striker, bowler)
    venue_avg = get_venue_avg(venue)

    aggression = vs_sr * form
    collapse_risk = wickets / (balls_left + 1)

    # =========================
    # MATCH CONTEXT
    # =========================
    powerplay = 1 if balls_bowled <= 36 else 0

    if balls_left <= 36:
        phase = 0
    elif balls_left <= 72:
        phase = 1
    else:
        phase = 2

    # =========================
    # FEATURE VECTOR
    # =========================
    features = np.array([[
        score,
        wickets,
        wickets_left,
        balls_left,
        required_rr,
        current_rr,
        pressure,

        runs_last_6,
        runs_last_12,
        momentum,
        acceleration,

        form,
        vs_sr,
        vs_avg,
        venue_avg,

        powerplay,
        aggression,
        phase,
        collapse_risk
    ]])

    features = scaler.transform(features)
    tensor = torch.tensor(features, dtype=torch.float32)

    # =========================
    # MODEL PREDICTION
    # =========================
    with torch.no_grad():
        base_prob = torch.sigmoid(model(tensor)).item()

    prob = calibrate_probability(
        base_prob=base_prob,
        target_runs=max(target_runs, 1),
        predict_overs=max(predict_overs, 1),
        current_rr=np.expm1(current_rr),
        wickets=wickets,
        venue_avg=venue_avg,
        form=form,
        vs_sr=vs_sr
    )

    prediction = "YES" if prob > 0.45 else "NO"

    # =========================
    # AGENT REASONING
    # =========================
    retrieval_result = retrieve_similar_cases(
        score=score,
        wickets=wickets,
        over=over,
        balls=balls,
        target_runs=target_runs,
        predict_overs=predict_overs,
        is_chasing=is_chasing,
        striker=striker,
        bowler=bowler,
        venue=venue
    )
    evidence_items = build_historical_evidence(
        striker=striker,
        bowler=bowler,
        venue=venue,
        target_runs=target_runs,
        predict_overs=predict_overs,
        is_chasing=is_chasing
    )

    prob = blend_with_historical_probability(prob, retrieval_result, predict_overs)
    prediction = "YES" if prob > 0.45 else "NO"
    retrieval_summary = build_retrieval_summary(retrieval_result, target_runs, predict_overs)
    historical_evidence = format_historical_evidence(evidence_items)

    context_label = "match chase context" if context_mode == "match_target" else "next-overs target context"
    innings_overs_left = max(total_overs - (over + balls / 6), 0)

    prompt = build_agent_prompt(
        score=score,
        wickets=wickets,
        over=over,
        balls=balls,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        venue=venue,
        target_total=target_total,
        remaining_runs=remaining_runs,
        innings_overs_left=innings_overs_left,
        predict_overs=predict_overs,
        context_label=context_label,
        current_rr_actual=np.expm1(current_rr),
        required_rr_actual=np.expm1(required_rr),
        target_runs=target_runs,
        retrieval_summary=retrieval_summary,
        historical_evidence=historical_evidence,
        prediction=prediction
    )

    analysis = build_fallback_analysis(
        score, wickets, over, balls, target_runs, predict_overs, striker, bowler, venue, prob
    )
    analysis += (
        f" Context used: {'match chase' if context_mode == 'match_target' else 'next-overs target'}."
    )
    analysis += f" {retrieval_summary}"

    if agent is not None:
        try:
            agent_res = agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            if agent_res and agent_res.get("messages"):
                analysis = (
                    f"Model verdict: {prediction} ({prob * 100:.1f}%).\n\n"
                    f"{agent_res['messages'][-1].content}"
                )
        except Exception:
            pass

    return jsonify({
        "probability": round(prob, 3),
        "prediction": prediction,
        "analysis": analysis,
        "historical_evidence": historical_evidence
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
