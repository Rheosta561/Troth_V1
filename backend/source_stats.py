import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _fetch_page_text(url):
    try:
        res = requests.get(url, timeout=6, headers={"User-Agent": USER_AGENT})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""


def _extract_float(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def _search_urls(query, allowed_domains, limit=5):
    urls = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=limit):
                href = item.get("href") or item.get("url") or ""
                if any(domain in href for domain in allowed_domains):
                    urls.append(href)
    except Exception:
        return []
    return urls


def _run_with_timeout(fn, timeout_seconds, fallback):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            return fallback
        except Exception:
            return fallback


@lru_cache(maxsize=256)
def fetch_batting_strike_rate_at_venue(player, venue):
    def lookup():
        query = f'"{player}" "{venue}" IPL strike rate site:espncricinfo.com OR site:cricbuzz.com'
        urls = _search_urls(query, ["espncricinfo.com", "cricbuzz.com"], limit=3)
        patterns = [
            r"strike rate[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)",
            r"sr[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)",
        ]

        for url in urls:
            text = _fetch_page_text(url)
            value = _extract_float(patterns, text)
            if value is not None:
                return {"value": value, "source": url}

        return {"value": None, "source": None}

    return _run_with_timeout(lookup, 2.5, {"value": None, "source": None})


@lru_cache(maxsize=256)
def fetch_bowler_economy_at_venue(bowler, venue):
    def lookup():
        query = f'"{bowler}" "{venue}" IPL economy site:espncricinfo.com OR site:cricbuzz.com'
        urls = _search_urls(query, ["espncricinfo.com", "cricbuzz.com"], limit=3)
        patterns = [
            r"economy[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)",
            r"econ[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)",
        ]

        for url in urls:
            text = _fetch_page_text(url)
            value = _extract_float(patterns, text)
            if value is not None:
                return {"value": value, "source": url}

        return {"value": None, "source": None}

    return _run_with_timeout(lookup, 2.5, {"value": None, "source": None})
