const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const PREDICT_URL = "http://127.0.0.1:5000/predict-live";

async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

function showError(message) {
  resultBox.classList.remove("hidden");
  document.getElementById("prediction").innerText = "Unavailable";
  document.getElementById("prob").innerText = "";
  document.getElementById("evidence").innerText = "";
  document.getElementById("analysis").innerText = message;
}

document.getElementById("submit").onclick = async () => {

  loader.classList.remove("hidden");
  resultBox.classList.add("hidden");

  const payload = {
    live: true,
    is_chasing: document.getElementById("is_chasing").checked,

    striker: val("striker"),
    non_striker: val("non_striker"),
    bowler: val("bowler"),

    score: num("score"),
    wickets: num("wickets"),
    over: num("over"),
    balls: num("balls"),
    venue: val("venue"),

    target_total: num("target_total"),
    total_overs: num("total_overs") || 20,

    target_runs: num("target_runs"),
    overs: num("overs")
  };

  if (payload.is_chasing && payload.target_total <= 0) {
    loader.classList.add("hidden");
    showError("Enter a target total when the team is chasing, or turn off the chasing toggle.");
    return;
  }

  try {
    const res = await fetchWithTimeout(PREDICT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    }, 12000);

    if (!res.ok) {
      throw new Error("Server error");
    }

    const json = await res.json();

    if (json.status === "no_live_match") {
      showError("No live match data was available for prediction.");
      return;
    }

    if (json.status === "invalid_input") {
      showError(json.message || "Please correct the match inputs and try again.");
      return;
    }

    resultBox.classList.remove("hidden");

    document.getElementById("prediction").innerText =
      json.prediction || "Unavailable";

    document.getElementById("prob").innerText =
      typeof json.probability === "number"
        ? "Probability: " + (json.probability * 100).toFixed(2) + "%"
        : "";

    document.getElementById("evidence").innerText =
      json.historical_evidence || "No historical evidence was available for this scenario.";

    document.getElementById("analysis").innerText =
      json.analysis || "Prediction completed, but no analysis text was returned.";

  } catch (err) {
    console.log(err);
    showError("Could not reach the backend in time. Check that the Flask server is running on port 5000.");
  } finally {
    loader.classList.add("hidden");
  }
};

// helpers
function val(id) {
  return document.getElementById(id).value;
}

function num(id) {
  return parseInt(val(id)) || 0;
}
