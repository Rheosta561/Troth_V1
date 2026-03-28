import requests
import re

# =========================
# FETCH COMMENTARY (ESPN)
# =========================
def fetch_commentary():
    try:
        url = "https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/scoreboard"
        data = requests.get(url, timeout=3).json()

        events = data.get("events", [])
        if not events:
            return []

        live = next((e for e in events if e["status"]["type"]["state"] == "in"), None)
        if not live:
            return []

        comp = live["competitions"][0]

        # commentary endpoint
        match_id = live["id"]
        comm_url = f"https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/playbyplay?event={match_id}"

        comm_data = requests.get(comm_url, timeout=3).json()

        plays = comm_data.get("plays", [])

        return plays[-30:]  # last ~30 balls

    except:
        return []
def extract_runs_and_players(plays):
    runs = []
    striker = "unknown"

    for p in reversed(plays):

        text = p.get("text", "").lower()

        # extract runs
        if "four" in text:
            runs.append(4)
        elif "six" in text:
            runs.append(6)
        else:
            match = re.search(r"(\d+) run", text)
            if match:
                runs.append(int(match.group(1)))
            elif "no run" in text:
                runs.append(0)

        # extract striker name
        if "to" in text:
            parts = text.split("to")
            if len(parts) > 1:
                striker = parts[1].split(",")[0].strip()

        if len(runs) >= 12:
            break

    return runs[:12], striker
def compute_momentum():
    plays = fetch_commentary()

    if not plays:
        return 10, 20, "unknown"

    runs, striker = extract_runs_and_players(plays)

    runs_last_6 = sum(runs[:6]) if len(runs) >= 6 else sum(runs)
    runs_last_12 = sum(runs)

    return runs_last_6, runs_last_12, striker
