const status = document.getElementById("status");
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

document.getElementById("manualBtn").onclick = () => {
  chrome.tabs.create({ url: "form.html" });
};

document.getElementById("liveBtn").onclick = async () => {

  status.innerText = "Fetching match...";
  resultBox.classList.add("hidden");

  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});

  chrome.tabs.sendMessage(tab.id, {type: "GET_DATA"}, async (data) => {

    if (!data || !data.live) {
      status.innerText = "❌ No Live Match";
      return;
    }

    status.innerText = "Live match detected ✅";

    // ask only missing inputs
    data.non_striker = prompt("Non-striker:");

    data.target_runs = parseInt(prompt("Runs to predict:"));
    data.overs = parseInt(prompt("Overs to predict:"));

    data.is_chasing = confirm("Is chasing?");

    if (data.is_chasing) {
      data.target_total = parseInt(prompt("Target total:")) || 0;
    } else {
      data.target_total = 0;
    }

    runPrediction(data);
  });
};

async function runPrediction(payload) {

  loader.classList.remove("hidden");
  status.innerText = "";

  try {
    const res = await fetchWithTimeout(PREDICT_URL, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
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
}
