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
  srh: {
    batters: [
      "Ishan Kishan", "Aniket Verma", "Smaran Ravichandran", "Salil Arora",
      "Heinrich Klaasen", "Travis Head"
    ],
    allRounders: [
      "Harshal Patel", "Kamindu Mendis", "Harsh Dubey", "Brydon Carse",
      "Shivang Kumar", "Krains Fuletra", "Liam Livingstone", "David Payne",
      "Abhishek Sharma", "Nitish Kumar Reddy"
    ],
    bowlers: [
      "Pat Cummins", "Zeeshan Ansari", "Jaydev Unadkat", "Eshan Malinga",
      "Sakib Hussain", "Onkar Tarmale", "Amit Kumar", "Praful Hinge", "Shivam Mavi"
    ]
  },
  "sunrisers hyderabad": {
    batters: [
      "Ishan Kishan", "Aniket Verma", "Smaran Ravichandran", "Salil Arora",
      "Heinrich Klaasen", "Travis Head"
    ],
    allRounders: [
      "Harshal Patel", "Kamindu Mendis", "Harsh Dubey", "Brydon Carse",
      "Shivang Kumar", "Krains Fuletra", "Liam Livingstone", "David Payne",
      "Abhishek Sharma", "Nitish Kumar Reddy"
    ],
    bowlers: [
      "Pat Cummins", "Zeeshan Ansari", "Jaydev Unadkat", "Eshan Malinga",
      "Sakib Hussain", "Onkar Tarmale", "Amit Kumar", "Praful Hinge", "Shivam Mavi"
    ]
  },
  kkr: {
    batters: [
      "Ajinkya Rahane", "Rinku Singh", "Angkrish Raghuvanshi", "Manish Pandey",
      "Finn Allen", "Tejasvi Singh", "Rahul Tripathi", "Tim Seifert", "Rovman Powell"
    ],
    allRounders: [
      "Anukul Roy", "Cameron Green", "Sarthak Ranjan", "Daksh Kamra",
      "Rachin Ravindra", "Ramandeep Singh", "Sunil Narine"
    ],
    bowlers: [
      "Blessing Muzarabani", "Vaibhav Arora", "Matheesha Pathirana", "Kartik Tyagi",
      "Prashant Solanki", "Saurabh Dubey", "Navdeep Saini", "Umran Malik",
      "Varun Chakaravarthy"
    ]
  },
  "kolkata knight riders": {
    batters: [
      "Ajinkya Rahane", "Rinku Singh", "Angkrish Raghuvanshi", "Manish Pandey",
      "Finn Allen", "Tejasvi Singh", "Rahul Tripathi", "Tim Seifert", "Rovman Powell"
    ],
    allRounders: [
      "Anukul Roy", "Cameron Green", "Sarthak Ranjan", "Daksh Kamra",
      "Rachin Ravindra", "Ramandeep Singh", "Sunil Narine"
    ],
    bowlers: [
      "Blessing Muzarabani", "Vaibhav Arora", "Matheesha Pathirana", "Kartik Tyagi",
      "Prashant Solanki", "Saurabh Dubey", "Navdeep Saini", "Umran Malik",
      "Varun Chakaravarthy"
    ]
  }
};

const TEAM_PLAYERS = {
  srh: [
    "Top order: Travis Head, Abhishek Sharma, Ishan Kishan, Heinrich Klaasen, Aiden Markram",
    "Middle/death hitters: Nitish Kumar Reddy, Abdul Samad, Liam Livingstone, Aniket Verma",
    "Spinners: Zeeshan Ansari, Kamindu Mendis, Harsh Dubey",
    "Pacers: Pat Cummins, Jaydev Unadkat, Eshan Malinga, Shivam Mavi, Harshal Patel, Brydon Carse"
  ],
  "sunrisers hyderabad": [
    "Top order: Travis Head, Abhishek Sharma, Ishan Kishan, Heinrich Klaasen, Aiden Markram",
    "Middle/death hitters: Nitish Kumar Reddy, Abdul Samad, Liam Livingstone, Aniket Verma",
    "Spinners: Zeeshan Ansari, Kamindu Mendis, Harsh Dubey",
    "Pacers: Pat Cummins, Jaydev Unadkat, Eshan Malinga, Shivam Mavi, Harshal Patel, Brydon Carse"
  ],
  kkr: [
    "Top order: Sunil Narine, Ajinkya Rahane, Angkrish Raghuvanshi, Finn Allen, Tim Seifert",
    "Middle/death hitters: Rinku Singh, Rovman Powell, Ramandeep Singh, Cameron Green",
    "Spinners: Sunil Narine, Varun Chakaravarthy, Anukul Roy, Prashant Solanki",
    "Pacers: Vaibhav Arora, Kartik Tyagi, Matheesha Pathirana, Blessing Muzarabani, Navdeep Saini, Umran Malik"
  ],
  "kolkata knight riders": [
    "Top order: Sunil Narine, Ajinkya Rahane, Angkrish Raghuvanshi, Finn Allen, Tim Seifert",
    "Middle/death hitters: Rinku Singh, Rovman Powell, Ramandeep Singh, Cameron Green",
    "Spinners: Sunil Narine, Varun Chakaravarthy, Anukul Roy, Prashant Solanki",
    "Pacers: Vaibhav Arora, Kartik Tyagi, Matheesha Pathirana, Blessing Muzarabani, Navdeep Saini, Umran Malik"
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
