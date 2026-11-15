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
from urllib.parse import quote, quote_plus, urlsplit

import httpx

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
    "spotify": ("https://open.spotify.com", "https://open.spotify.com/search/{q}"),
}
_PATH_QUOTED = frozenset({"spotify"})     # the query is a path segment here, so spaces are %20, not +
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
        built = search.format(q=quote(query, safe="") if key in _PATH_QUOTED else quote_plus(query))
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


_VIDEO_ID = re.compile(r'"videoRenderer":\{"videoId":"([A-Za-z0-9_-]{11})"')
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def first_video_id(page: str) -> str | None:
    """The id of the first ordinary video in a YouTube results page (not an ad, a channel or a Short), or None."""
    m = _VIDEO_ID.search(page)
    return m.group(1) if m else None


def watch_url(video_id: str) -> str | None:
    """The watch page for an id, or None unless it is exactly 11 safe characters. The only address EV builds from page text."""
    return f"https://www.youtube.com/watch?v={video_id}" if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None


def resolve_first_video(query: str, timeout_s: float = 6.0) -> str | None:
    """The watch URL of the top YouTube result for `query`, or None if the page can't be fetched or has no video.

    This is the one place EV reads a page from youtube.com itself. Nothing from the page is used except an 11-character
    video id that has passed watch_url's check; titles and descriptions are never read, spoken or shown to the model."""
    q = re.sub(r"\s+", " ", query).strip()
    if not q:
        return None
    try:
        r = httpx.get(SITES["youtube"][1].format(q=quote_plus(q)), timeout=timeout_s, follow_redirects=False,
                      headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    vid = first_video_id(r.text)
    return watch_url(vid) if vid else None


def host_of(url: str) -> str:
    return urlsplit(url).hostname or ""


_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,59}")


def open_url(url: str, browser: str = "", profile: str = "") -> str:
    """Open `url` in the named browser (new tab if it's running), or the system default. Returns what was used.

    `profile` is a Firefox profile name from profiles.ini and only comes from config ([web] media_profile), never from the
    user's words or the model. Without it Windows hands the link to whichever Firefox is running, which may be the wrong profile."""
    key = browser.strip().lower()
    if key:
        exe = BROWSERS.get(key)
        path = system.resolve_exe(exe) if exe else None
        if not path:
            raise FileNotFoundError(f"I couldn't find {browser} on this PC")
        argv = [path, url]
        if profile and exe == "firefox.exe":
            if not _PROFILE_NAME.fullmatch(profile):
                raise ValueError("bad browser profile name")
            argv = [path, "-P", profile, url]
        subprocess.Popen(argv, close_fds=True)  # noqa: S603  # fixed browser exe, URL validated by build_url, profile from config
        return key
    if os.name == "nt":
        os.startfile(url)  # type: ignore[attr-defined]  # noqa: S606
    elif not webbrowser.open(url):
        raise OSError("no browser available")
    return "your browser"
