"""Open websites and site searches in a browser. URLs are built here, never taken on trust from the model.

`build_url` is a pure function of its arguments, so the safety policy can call it to judge the exact URL that
would be opened, using the same code the tool uses.
"""
from __future__ import annotations

import ipaddress
import os
import re
import subprocess  # noqa: S404
import webbrowser
from urllib.parse import quote_plus, urlsplit

from . import system

# site -> (home page, search URL with {q})
SITES: dict[str, tuple[str, str]] = {
    "youtube": ("https://www.youtube.com", "https://www.youtube.com/results?search_query={q}"),
    "google": ("https://www.google.com", "https://www.google.com/search?q={q}"),
    "gmail": ("https://mail.google.com", "https://mail.google.com/mail/u/0/#search/{q}"),
    "maps": ("https://www.google.com/maps", "https://www.google.com/maps/search/{q}"),
    "wikipedia": ("https://en.wikipedia.org", "https://en.wikipedia.org/w/index.php?search={q}"),
    "github": ("https://github.com", "https://github.com/search?q={q}"),
    "reddit": ("https://www.reddit.com", "https://www.reddit.com/search/?q={q}"),
    "stackoverflow": ("https://stackoverflow.com", "https://stackoverflow.com/search?q={q}"),
    "amazon": ("https://www.amazon.in", "https://www.amazon.in/s?k={q}"),
    "twitter": ("https://x.com", "https://x.com/search?q={q}"),
    "linkedin": ("https://www.linkedin.com", "https://www.linkedin.com/search/results/all/?keywords={q}"),
    "duckduckgo": ("https://duckduckgo.com", "https://duckduckgo.com/?q={q}"),
}
SITE_ALIASES = {
    "you tube": "youtube", "yt": "youtube", "google maps": "maps", "g mail": "gmail", "google mail": "gmail",
    "x": "twitter", "stack overflow": "stackoverflow", "git hub": "github", "duck duck go": "duckduckgo",
    "wiki": "wikipedia",
}
BROWSERS = {
    "firefox": "firefox.exe", "chrome": "chrome.exe", "google chrome": "chrome.exe", "edge": "msedge.exe",
    "microsoft edge": "msedge.exe", "brave": "brave.exe",
}
KNOWN_HOSTS = {urlsplit(home).hostname for home, _ in SITES.values()} | {
    urlsplit(search).hostname for _, search in SITES.values()}

_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PRIVATE_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet")
_LATEST = re.compile(r"\b(latest|newest|most recent|recent|new)\b", re.I)
YOUTUBE_NEWEST = "&sp=CAI%3D"          # YouTube's "sort by upload date" filter


def canonical_site(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    base = re.sub(r"\.(com|org|net|in)$", "", n)         # "youtube.com" -> "youtube", "example.com" stays
    if base in SITES or base in SITE_ALIASES:
        n = base
    return SITE_ALIASES.get(n, n)


def is_public_host(host: str | None) -> bool:
    """A normal public DNS name. No IP literals, localhost, intranet suffixes or userinfo tricks."""
    if not host or len(host) > 253:
        return False
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host.strip("[]"))
        return False
    except ValueError:
        pass
    if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES):
        return False
    labels = host.split(".")
    return len(labels) >= 2 and all(_LABEL.match(lab) for lab in labels) and labels[-1].isalpha() and len(labels[-1]) >= 2


def build_url(site: str = "", query: str = "", url: str = "") -> str | None:
    """The https URL to open, or None if the request isn't a safe public website."""
    query = re.sub(r"\s+", " ", query).strip()
    if url.strip():
        u = url.strip()
        if re.match(r"[a-z][a-z0-9+.-]*:", u, re.I) and not re.match(r"https?://", u, re.I):
            return None                               # javascript:, file:, data:, ftp:, mailto:, host:port ...
        if "://" not in u:
            u = "https://" + u
        try:
            parts = urlsplit(u)
            if parts.username or parts.password or parts.port is not None:
                return None
        except ValueError:                            # malformed netloc or port; never let policy code raise
            return None
        if not is_public_host(parts.hostname) or len(u) > 500 or re.search(r"\s", u):
            return None
        return "https://" + u.split("://", 1)[1]
    key = canonical_site(site)
    if key in SITES:
        home, search = SITES[key]
        if not query:
            return home
        built = search.format(q=quote_plus(query))
        return built + YOUTUBE_NEWEST if key == "youtube" and _LATEST.search(query) else built
    if site.strip():                                  # a bare domain like "example.com"
        host = key.replace(" ", "")
        if not is_public_host(host):
            return None
        if query:
            return SITES["google"][1].format(q=quote_plus(f"{query} site:{host}"))
        return "https://" + host
    if query:
        return SITES["google"][1].format(q=quote_plus(query))
    return None


def host_of(url: str) -> str:
    return urlsplit(url).hostname or ""


def open_url(url: str, browser: str = "") -> str:
    """Open `url` in the named browser (new tab if it's running), or the system default. Returns what was used."""
    key = browser.strip().lower()
    if key:
        exe = BROWSERS.get(key)
        path = system.resolve_exe(exe) if exe else None
        if not path:
            raise FileNotFoundError(f"I couldn't find {browser} on this PC")
        subprocess.Popen([path, url], close_fds=True)  # noqa: S603  # fixed browser exe, URL validated by build_url
        return key
    if os.name == "nt":
        os.startfile(url)  # type: ignore[attr-defined]  # noqa: S606
    elif not webbrowser.open(url):
        raise OSError("no browser available")
    return "your browser"
