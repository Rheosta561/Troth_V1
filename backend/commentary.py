import requests
import re

def fetch_commentary():
    try:
        url = "https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/scoreboard"
        data = requests.get(url).json()

        events = data.get("events", [])
        live = next((e for e in events if e["status"]["type"]["state"] == "in"), None)

        if not live:
            return []

        match_id = live["id"]

        comm_url = f"https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/playbyplay?event={match_id}"
        comm_data = requests.get(comm_url).json()

        return comm_data.get("plays", [])[-30:]

    except:
        return []

def extract_runs_and_players(plays):
    runs = []
    striker = "unknown"

    for p in reversed(plays):
        text = p.get("text","").lower()

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

        if "to" in text:
            try:
                striker = text.split("to")[1].split(",")[0].strip()
            except:
                pass

        if len(runs) >= 12:
            break

    return runs[:12], striker

def compute_momentum():
    plays = fetch_commentary()

    if not plays:
        return 10,20,"unknown"

    runs, striker = extract_runs_and_players(plays)

    return sum(runs[:6]), sum(runs), striker