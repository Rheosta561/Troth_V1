import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DDG_HTML_URL = "https://html.duckduckgo.com/html/"


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
    results = _search_duckduckgo(query, allowed_domains, limit=limit)
    return [item.get("source", "") for item in results if item.get("source")]


def _run_with_timeout(fn, timeout_seconds, fallback):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            return fallback
        except Exception:
            return fallback


def _clean_snippet(text):
    value = " ".join(str(text or "").split())
    return value[:220]


def _search_duckduckgo(query, allowed_domains=None, limit=5):
    allowed_domains = allowed_domains or []
    try:
        res = requests.get(
            DDG_HTML_URL,
            params={"q": query},
            timeout=3,
            headers={"User-Agent": USER_AGENT},
        )
        res.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    results = []

    for anchor in soup.select("a.result__a, a.result-link"):
        href = anchor.get("href") or ""
        if href.startswith("/"):
            href = urljoin(DDG_HTML_URL, href)

        if allowed_domains and not any(domain in href for domain in allowed_domains):
            continue

        container = anchor.find_parent(class_="result") or anchor.parent
        snippet_node = None
        if container is not None:
            snippet_node = container.select_one(".result__snippet, .result-snippet")

        title = _clean_snippet(anchor.get_text(" ", strip=True))
        snippet = _clean_snippet(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        if not href or not (title or snippet):
            continue

        results.append({
            "title": title,
            "snippet": snippet,
            "source": href,
        })

        if len(results) >= limit:
            break

    return results


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


@lru_cache(maxsize=256)
def fetch_search_context(striker, non_striker, bowler, venue, target_runs, predict_overs):
    def lookup():
        queries = [
            (
                f'"{striker}" "{venue}" IPL batting record strike rate',
                f"{striker} at {venue}"
            ),
            (
                f'"{non_striker}" "{venue}" IPL batting record strike rate',
                f"{non_striker} at {venue}"
            ),
            (
                f'"{striker}" "{bowler}" IPL record',
                f"{striker} vs {bowler}"
            ),
            (
                f'"{bowler}" "{venue}" IPL economy',
                f"{bowler} at {venue}"
            ),
            (
                f'"{venue}" IPL scoring pattern {int(target_runs)} runs {predict_overs} overs',
                f"{venue} scoring trend"
            ),
        ]

        evidence = []
        allowed_domains = ["espncricinfo.com", "cricbuzz.com", "iplt20.com"]
        for query, label in queries:
            try:
                results = _search_duckduckgo(query, allowed_domains=allowed_domains, limit=2)
            except Exception:
                continue

            for item in results:
                snippet = _clean_snippet(item.get("snippet") or "")
                href = item.get("source") or ""
                title = _clean_snippet(item.get("title") or label)
                if not snippet:
                    continue
                evidence.append({
                    "label": label,
                    "title": title,
                    "snippet": snippet,
                    "source": href,
                })
                break

        return evidence[:5]

    return _run_with_timeout(lookup, 2.0, [])
