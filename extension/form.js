const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const backendStatus = document.getElementById("backendStatus");
const battingTeamInput = document.getElementById("batting_team");
const bowlingTeamInput = document.getElementById("bowling_team");
const battingTeamPlayers = document.getElementById("batting_team_players");
const bowlingTeamPlayers = document.getElementById("bowling_team_players");
const strikerInput = document.getElementById("striker");
const nonStrikerInput = document.getElementById("non_striker");
const bowlerInput = document.getElementById("bowler");
const PREDICT_URL = "http://127.0.0.1:5055/predict-live";
const HEALTH_URL = "http://127.0.0.1:5055/health";

const TEAM_SQUADS = {
  gt: {
    batters: [
      "Shubman Gill", "Sai Sudharsan", "Jos Buttler", "Shahrukh Khan", "Kumar Kushagra"
    ],
    allRounders: [
      "Rahul Tewatia", "Washington Sundar", "Glenn Phillips", "Jason Holder"
    ],
    bowlers: [
      "Rashid Khan", "Kagiso Rabada", "Mohammed Siraj", "Prasidh Krishna",
      "Sai Kishore", "Ishant Sharma"
    ]
  },
  "gujarat titans": {
    batters: [
      "Shubman Gill", "Sai Sudharsan", "Jos Buttler", "Shahrukh Khan", "Kumar Kushagra"
    ],
    allRounders: [
      "Rahul Tewatia", "Washington Sundar", "Glenn Phillips", "Jason Holder"
    ],
    bowlers: [
      "Rashid Khan", "Kagiso Rabada", "Mohammed Siraj", "Prasidh Krishna",
      "Sai Kishore", "Ishant Sharma"
    ]
  },
  pbks: {
    batters: [
      "Shreyas Iyer", "Prabhsimran Singh", "Shashank Singh", "Nehal Wadhera", "Priyansh Arya"
    ],
    allRounders: [
      "Marcus Stoinis", "Marco Jansen", "Azmatullah Omarzai", "Harpreet Brar"
    ],
    bowlers: [
      "Arshdeep Singh", "Yuzvendra Chahal", "Lockie Ferguson", "Nathan Ellis"
    ]
  },
  "punjab kings": {
    batters: [
      "Shreyas Iyer", "Prabhsimran Singh", "Shashank Singh", "Nehal Wadhera", "Priyansh Arya"
    ],
    allRounders: [
      "Marcus Stoinis", "Marco Jansen", "Azmatullah Omarzai", "Harpreet Brar"
    ],
    bowlers: [
      "Arshdeep Singh", "Yuzvendra Chahal", "Lockie Ferguson", "Nathan Ellis"
    ]
  }
};

const TEAM_PLAYERS = {
  gt: [
    "Shubman Gill", "Sai Sudharsan", "Jos Buttler", "Shahrukh Khan", "Kumar Kushagra",
    "Rahul Tewatia", "Washington Sundar", "Glenn Phillips", "Jason Holder",
    "Rashid Khan", "Kagiso Rabada", "Mohammed Siraj", "Prasidh Krishna",
    "Sai Kishore", "Ishant Sharma"
  ],
  "gujarat titans": [
    "Shubman Gill", "Sai Sudharsan", "Jos Buttler", "Shahrukh Khan", "Kumar Kushagra",
    "Rahul Tewatia", "Washington Sundar", "Glenn Phillips", "Jason Holder",
    "Rashid Khan", "Kagiso Rabada", "Mohammed Siraj", "Prasidh Krishna",
    "Sai Kishore", "Ishant Sharma"
  ],
  pbks: [
    "Shreyas Iyer", "Prabhsimran Singh", "Shashank Singh", "Nehal Wadhera", "Priyansh Arya",
    "Marcus Stoinis", "Marco Jansen", "Azmatullah Omarzai", "Harpreet Brar",
    "Arshdeep Singh", "Yuzvendra Chahal", "Lockie Ferguson", "Nathan Ellis"
  ],
  "punjab kings": [
    "Shreyas Iyer", "Prabhsimran Singh", "Shashank Singh", "Nehal Wadhera", "Priyansh Arya",
    "Marcus Stoinis", "Marco Jansen", "Azmatullah Omarzai", "Harpreet Brar",
    "Arshdeep Singh", "Yuzvendra Chahal", "Lockie Ferguson", "Nathan Ellis"
  ]
};

async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function parseResponse(res) {
  const text = await res.text();

  try {
    return text ? JSON.parse(text) : {};
  } catch (err) {
    return {
      status: "invalid_response",
      message: text || "Backend returned a non-JSON response."
    };
  }
}

async function checkBackendHealth() {
  if (!backendStatus) {
    return false;
  }

  backendStatus.textContent = "Checking backend connection...";
  backendStatus.className = "backend-status";

  try {
    const res = await fetchWithTimeout(HEALTH_URL, {}, 3000);
    const json = await parseResponse(res);

    if (!res.ok || json.status !== "ok") {
      backendStatus.textContent = "Backend offline. Start Flask on port 5055.";
      backendStatus.className = "backend-status backend-status--error";
      return false;
    }

    backendStatus.textContent = "Backend connected on port 5055.";
    backendStatus.className = "backend-status backend-status--ok";
    return true;
  } catch (err) {
    backendStatus.textContent = "Backend offline. Start Flask on port 5055.";
    backendStatus.className = "backend-status backend-status--error";
    return false;
  }
}

function showError(message) {
  resultBox.classList.remove("hidden");
  document.getElementById("prediction").innerText = "Unavailable";
  document.getElementById("prob").innerText = "";
  document.getElementById("evidence").innerText = "";
  document.getElementById("analysis").innerText = message;
}

function normalizeTeamKey(value) {
  return String(value || "").trim().toLowerCase();
}

function renderTeamPlayers() {
  const battingKey = normalizeTeamKey(battingTeamInput?.value);
  const bowlingKey = normalizeTeamKey(bowlingTeamInput?.value);
  const battingList = TEAM_PLAYERS[battingKey] || [];
  const bowlingList = TEAM_PLAYERS[bowlingKey] || [];

  if (battingTeamPlayers) {
    battingTeamPlayers.textContent = battingList.length
      ? battingList.join(" • ")
      : "No local player guide found for this team yet.";
  }

  if (bowlingTeamPlayers) {
    bowlingTeamPlayers.textContent = bowlingList.length
      ? bowlingList.join(" • ")
      : "No local player guide found for this team yet.";
  }
}

function setSelectOptions(selectEl, label, items) {
  if (!selectEl) {
    return;
  }

  const currentValue = selectEl.value;
  selectEl.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = label;
  selectEl.appendChild(placeholder);

  for (const item of items) {
    const option = document.createElement("option");
    option.value = item;
    option.textContent = item;
    selectEl.appendChild(option);
  }

  if (items.includes(currentValue)) {
    selectEl.value = currentValue;
  }
}

function buildBattingOptions(teamKey) {
  const squad = TEAM_SQUADS[teamKey];
  if (!squad) {
    return [];
  }
  return [...squad.batters, ...squad.allRounders];
}

function buildBowlingOptions(teamKey) {
  const squad = TEAM_SQUADS[teamKey];
  if (!squad) {
    return [];
  }
  return [...squad.bowlers, ...squad.allRounders];
}

function renderPlayerDropdowns() {
  const battingKey = normalizeTeamKey(battingTeamInput?.value);
  const bowlingKey = normalizeTeamKey(bowlingTeamInput?.value);

  setSelectOptions(strikerInput, "Striker", buildBattingOptions(battingKey));
  setSelectOptions(nonStrikerInput, "Non-striker", buildBattingOptions(battingKey));
  setSelectOptions(bowlerInput, "Bowler", buildBowlingOptions(bowlingKey));
}

document.getElementById("submit").onclick = async () => {
  loader.classList.remove("hidden");
  resultBox.classList.add("hidden");

  const payload = {
    live: false,
    source: "manual",
    is_chasing: document.getElementById("is_chasing").checked,

    batting_team: val("batting_team"),
    bowling_team: val("bowling_team"),
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

  if (payload.overs <= payload.over || (payload.overs === payload.over && payload.balls > 0)) {
    loader.classList.add("hidden");
    showError("Target end over must be later than the current over position.");
    return;
  }

  if (payload.total_overs > 0 && payload.overs > payload.total_overs) {
    loader.classList.add("hidden");
    showError("Target end over cannot be greater than the innings total overs.");
    return;
  }

  if (payload.target_runs <= payload.score) {
    loader.classList.add("hidden");
    showError("Score target by end over must be greater than the current score.");
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

    const json = await parseResponse(res);

    if (!res.ok) {
      showError(json.message || `Prediction request failed (HTTP ${res.status}).`);
      return;
    }

    if (json.status === "no_live_match") {
      showError(json.message || "No live match data was available for prediction.");
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
    showError("Could not reach the backend in time. Check that Flask is running on port 5055.");
  } finally {
    loader.classList.add("hidden");
    checkBackendHealth();
  }
};

function val(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}

function num(id) {
  return parseInt(val(id), 10) || 0;
}

checkBackendHealth();
renderTeamPlayers();
renderPlayerDropdowns();
battingTeamInput?.addEventListener("input", renderTeamPlayers);
bowlingTeamInput?.addEventListener("input", renderTeamPlayers);
battingTeamInput?.addEventListener("input", renderPlayerDropdowns);
bowlingTeamInput?.addEventListener("input", renderPlayerDropdowns);
