from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


ESPN_SCOREBOARD_URL = "https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/scoreboard"
CRICBUZZ_LIVE_URL = "https://www.cricbuzz.com/cricket-match/live-scores"


def norm(value):
    return str(value).lower().strip()


def normalize_team_name(name):
    value = norm(name)
    replacements = {
        "mumbai indians": ["mi", "mumbai indians"],
        "kolkata knight riders": ["kkr", "kolkata knight riders"],
        "royal challengers bengaluru": ["rcb", "royal challengers bangalore", "royal challengers bengaluru"],
        "sunrisers hyderabad": ["srh", "sunrisers hyderabad"],
    }
    for canonical, aliases in replacements.items():
        if value in aliases:
            return canonical
    return value


def fetch_live_match_from_espn(batting_team, bowling_team):
    try:
        res = requests.get(ESPN_SCOREBOARD_URL, timeout=4)
        res.raise_for_status()
        data = res.json()

        batting_norm = normalize_team_name(batting_team)
        bowling_norm = normalize_team_name(bowling_team)

        for event in data.get("events", []):
            if event.get("status", {}).get("type", {}).get("state") != "in":
                continue

            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            team_names = {
                normalize_team_name(team.get("team", {}).get("displayName", ""))
                for team in competitors
            }

            if batting_norm in team_names and bowling_norm in team_names:
                return {
                    "provider": "espn",
                    "live": True,
                    "event_id": event.get("id"),
                    "venue": competition.get("venue", {}).get("fullName", ""),
                    "status": event.get("status", {}).get("type", {}).get("detail", ""),
                }
    except Exception:
        return None

    return None


def fetch_live_match_from_cricbuzz(batting_team, bowling_team):
    try:
        res = requests.get(CRICBUZZ_LIVE_URL, timeout=4, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        html = res.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True).lower()

        batting_norm = normalize_team_name(batting_team)
        bowling_norm = normalize_team_name(bowling_team)

        patterns = [batting_norm, bowling_norm]
        if all(re.search(re.escape(pattern), text) for pattern in patterns):
            return {
                "provider": "cricbuzz",
                "live": True,
                "event_id": None,
                "venue": "",
                "status": "Live match card matched on Cricbuzz",
            }
    except Exception:
        return None

    return None


def fetch_live_match_context(batting_team, bowling_team):
    espn_match = fetch_live_match_from_espn(batting_team, bowling_team)
    cricbuzz_match = fetch_live_match_from_cricbuzz(batting_team, bowling_team)

    if espn_match or cricbuzz_match:
        return {
            "espn": espn_match,
            "cricbuzz": cricbuzz_match,
            "live": True,
        }

    return {
        "espn": None,
        "cricbuzz": None,
        "live": False,
    }
