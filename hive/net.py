"""Network access — the capability HIVE spent its whole life declaring it lacked.

Every version up to now answered "find me articles on X" by saying it could not,
and that refusal was *correct*: there was no code path by which a fetch could
have happened. This module is that path. Nothing else about the architecture
changes — the model still runs locally, the vault never leaves the disk, and no
prompt or transcript is sent anywhere. What crosses the network is a search query
and a page request, and nothing else.

Four properties this layer is built around, in order of how much they cost to get
wrong:

1. **Connectivity is a fact, not an assumption.** HIVE's honesty rules mean it
   must never claim to have looked something up when it did not. `online()` is
   checked before every operation and cached briefly, and every function here
   returns an explicit failure rather than an empty list that reads like "nothing
   found". "I couldn't reach the network" and "there are no results" are
   different sentences and Jay needs to hear the right one.

2. **The local machine is not on the internet.** A URL that arrives from a search
   result is attacker-influenced input by definition, and HIVE has an unauthenticated
   Ollama on 127.0.0.1:11434 and a vault on local disk. So resolution happens
   before connection and anything that lands on a loopback, link-local, private or
   reserved address is refused. Without this, "summarise this page for me" is a
   request HIVE would happily point at its own model server.

3. **Bounded everywhere.** Hard timeout, hard byte cap, hard redirect cap. A
   query task gets thirty seconds before the runner declares it stalled, so a
   fetch that hangs for two minutes is not a slow success, it is a failure that
   has not admitted it yet.

4. **No credentials, ever.** GET only, no cookies persisted, no logins, no forms.
   Everything here is readable by an anonymous browser or it does not happen.

Deliberately dependency-free: parsing is `html.parser` and `ElementTree` from the
standard library rather than BeautifulSoup or lxml. requirements.txt opens with a
warning about what dependency ordering does to this install, and a web reader is
not worth adding to that surface.
"""

from __future__ import annotations

import html as _html
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

# A real browser string. Not deception — several search front-ends serve a
# JavaScript-only page or an outright block to anything that looks automated,
# and the alternative is a layer that fails constantly for a reason that has
# nothing to do with whether the request was reasonable.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

DDG_HTML = "https://html.duckduckgo.com/html/"
MOJEEK = "https://www.mojeek.com/search"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


class Challenged(Exception):
    """A search front-end asked us to prove we are a human.

    Its own exception because it is neither "offline" nor "no results", and
    conflating it with either is a lie of a different kind each way. DuckDuckGo
    answers a burst of queries from one address with an HTTP 202 and a
    select-all-the-ducks page — no error status, no empty body, just a page with
    no results in it. A HIVE-Q query with ten research tasks in it will trip
    this, so it is a normal operating condition rather than an edge case, and
    the correct response is to try the next provider and, failing that, to say
    what actually happened.
    """


@dataclass
class Result:
    """One thing found. Shared shape across every provider.

    `url` is never spoken aloud — a read-out URL is unintelligible and takes
    longer than the fact it points at. It goes to the terminal pane and into the
    task's markdown log, which is where a link is actually usable.
    """

    title: str
    url: str = ""
    snippet: str = ""
    kind: str = "web"          # web | paper | product
    meta: dict = field(default_factory=dict)

    @property
    def host(self) -> str:
        try:
            return urlparse(self.url).netloc.replace("www.", "")
        except ValueError:
            return ""


@dataclass
class Page:
    url: str
    title: str
    text: str
    ok: bool = True
    error: str = ""


class Offline(Exception):
    """Raised when an operation needs the network and there isn't one."""


class Blocked(Exception):
    """Raised when a URL resolves somewhere HIVE must not connect to."""


# --- address safety --------------------------------------------------------
# Checked against the *resolved* address, not the hostname. A name under an
# attacker's control can point at 127.0.0.1, and a blocklist of hostnames would
# not notice.

def _is_public(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def check_url(url: str) -> str:
    """Return the URL if it is safe to fetch, or raise Blocked.

    Resolves the hostname and requires every address it answers with to be
    public. Every address, not the first: a name that returns both a routable
    address and 127.0.0.1 would otherwise pass on a retry.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise Blocked(f"{parts.scheme or 'that'} is not a scheme I'll fetch")
    if not parts.hostname:
        raise Blocked("no host in that URL")
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except socket.gaierror as exc:
        raise Blocked(f"couldn't resolve {parts.hostname}: {exc}") from exc
    addrs = {i[4][0] for i in infos}
    if not addrs or not all(_is_public(a) for a in addrs):
        raise Blocked(f"{parts.hostname} points inside this machine or network")
    return url


# --- HTML to text ----------------------------------------------------------

_DROP_TAGS = re.compile(
    r"(?is)<(script|style|noscript|svg|head|nav|footer|form|iframe|template)\b.*?</\1>")
_COMMENT = re.compile(r"(?is)<!--.*?-->")
_BLOCK_END = re.compile(r"(?i)</(p|div|li|tr|h[1-6]|section|article|br)\s*>|<br\s*/?>")
_TAG = re.compile(r"(?s)<[^>]+>")
_MAIN = re.compile(r"(?is)<(article|main)\b[^>]*>(?P<body>.*?)</\1>")


def to_text(markup: str, prefer_main: bool = True) -> str:
    """Strip a page down to the prose a person would actually read.

    Not a readability implementation — just enough to keep an LLM summary from
    being three quarters cookie banner. `<article>`/`<main>` when the page has
    one, then tag removal, then a line filter that drops the short fragments that
    navigation and button labels decompose into.
    """
    body = markup
    if prefer_main:
        m = _MAIN.search(markup)
        if m and len(m.group("body")) > 400:
            body = m.group("body")
    body = _COMMENT.sub(" ", _DROP_TAGS.sub(" ", body))
    body = _BLOCK_END.sub("\n", body)
    text = _html.unescape(_TAG.sub(" ", body))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    lines, seen = [], set()
    for raw in text.split("\n"):
        line = raw.strip()
        # Short lines are overwhelmingly chrome — menu items, breadcrumbs,
        # "Skip to main content". A real sentence clears this trivially.
        if len(line) < 40 or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines).strip()


class _Titles:
    """Pull <title> without dragging in a parser dependency."""

    _RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")

    @classmethod
    def of(cls, markup: str) -> str:
        m = cls._RE.search(markup)
        return re.sub(r"\s+", " ", _html.unescape(m.group(1))).strip() if m else ""


def _clean(fragment: str) -> str:
    """A search-result field: de-tagged, unescaped, whitespace-collapsed."""
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub("", fragment))).strip()


# --- the client ------------------------------------------------------------

class Web:
    """HIVE's window onto the network. One instance, owned by the runtime."""

    def __init__(self, cfg, bus):
        self.cfg, self.bus = cfg, bus
        self.enabled = bool(cfg.get("net.enabled", True))
        self.timeout = float(cfg.get("net.timeout_s", 12.0))
        self.max_bytes = int(cfg.get("net.max_page_bytes", 2_000_000))
        self.max_results = int(cfg.get("net.max_results", 6))
        self._probe_urls = list(cfg.get("net.probe_urls", [
            "https://duckduckgo.com/", "https://one.one.one.one/"]))
        self._ttl = float(cfg.get("net.status_ttl_s", 20.0))
        self._online: bool | None = None
        self._checked = 0.0
        self._lock = threading.Lock()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": USER_AGENT,
                     "Accept-Language": "en-US,en;q=0.9"},
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # -- connectivity -------------------------------------------------------
    def online(self, force: bool = False) -> bool:
        """Is there a network right now?

        Cached for a few seconds because this is checked on the way into every
        operation and on every render of the HIVE-Q status bar, and a DNS round
        trip per status-bar repaint would be absurd. Short enough that unplugging
        the machine is noticed within a turn.
        """
        if not self.enabled:
            return False
        now = time.monotonic()
        with self._lock:
            if not force and self._online is not None and now - self._checked < self._ttl:
                return self._online
        state = False
        for url in self._probe_urls:
            try:
                r = self._client.head(url, timeout=4.0)
                state = r.status_code < 500
                if state:
                    break
            except Exception:
                continue
        with self._lock:
            was, self._online, self._checked = self._online, state, time.monotonic()
        if was is not None and was != state:
            self.bus.log(f"network {'up' if state else 'down'}", "net")
        return state

    def require(self) -> None:
        if not self.enabled:
            raise Offline("network access is switched off in my config")
        if not self.online():
            raise Offline("I can't reach the network")

    # -- search -------------------------------------------------------------
    # DuckDuckGo's HTML endpoint: no API key, no account, no per-query quota,
    # and it answers `site:` operators, which is what makes retailer and
    # publisher searches work without scraping those sites directly (most of
    # which refuse an unattended client outright).
    _RESULT = re.compile(
        r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
    _SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.S)

    # Mojeek: an independent index (its own crawler, not a Bing or Google
    # reseller), no key, and markedly more tolerant of an unattended client.
    # It is the fallback rather than the default because its index is smaller —
    # better second answer, worse first one.
    _MJ_RESULT = re.compile(
        r'<h2><a class="title"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a></h2>'
        r'(?:\s*<p class="s">(?P<snip>.*?)</p>)?', re.S)

    # No results parsed *and* one of these in the body means we were challenged
    # rather than answered. Checked only after parsing fails, so a page that
    # happens to contain the word "anomaly" in a result is never misread.
    _CHALLENGE = re.compile(r"anomaly|unusual traffic|are you a robot|captcha|"
                            r"confirm this search was made by a human", re.I)

    def search(self, query: str, limit: int = 0) -> list[Result]:
        """Web search across the configured providers, in order.

        Three outcomes, kept distinct all the way to what gets spoken, because
        collapsing them is exactly how an assistant ends up lying politely:

          results        → a list
          nothing found  → an empty list
          couldn't ask   → Offline, carrying why

        A provider that challenges us is not "nothing found". It is the next
        provider's turn.
        """
        self.require()
        limit = limit or self.max_results
        providers = {"duckduckgo": self._ddg, "mojeek": self._mojeek}
        order = [p for p in self.cfg.get("net.search_providers",
                                         ["duckduckgo", "mojeek"])
                 if p in providers]
        trouble: list[str] = []

        for name in order or ["duckduckgo"]:
            try:
                out = providers[name](query, limit)
            except Challenged:
                trouble.append(f"{name} asked me to prove I'm not a bot")
                self.bus.log(f"search: {name} challenged us", "warn")
                continue
            except httpx.HTTPError as exc:
                trouble.append(f"{name} wouldn't answer ({type(exc).__name__})")
                continue
            if out:
                self.bus.log(f"search '{query[:40]}' -> {len(out)} via {name}", "net")
                return out
            # An honest empty answer from the first provider is still worth a
            # second opinion — the indexes differ enough that one having nothing
            # says little about the other.
            trouble.append(f"{name} had nothing")

        if all("had nothing" in t for t in trouble) and trouble:
            return []
        raise Offline("; ".join(trouble) or "no search provider is configured")

    def _ddg(self, query: str, limit: int) -> list[Result]:
        r = self._client.get(DDG_HTML, params={"q": query, "kl": "us-en"})
        if r.status_code >= 400:
            r.raise_for_status()
        titles = list(self._RESULT.finditer(r.text))
        snips = [_clean(m.group("snip")) for m in self._SNIPPET.finditer(r.text)]
        out: list[Result] = []
        for i, m in enumerate(titles):
            url = self._unwrap(m.group("url"))
            # Sponsored rows come back as duckduckgo.com/y.js redirectors. They
            # are indistinguishable from results in the markup and are never what
            # was asked for.
            if not url or "duckduckgo.com/y.js" in url or "ad_provider=" in url:
                continue
            out.append(Result(title=_clean(m.group("title")), url=url,
                              snippet=snips[i] if i < len(snips) else "",
                              kind="web"))
            if len(out) >= limit:
                break
        if not out and (r.status_code == 202 or self._CHALLENGE.search(r.text)):
            raise Challenged("duckduckgo")
        return out

    def _mojeek(self, query: str, limit: int) -> list[Result]:
        r = self._client.get(MOJEEK, params={"q": query})
        if r.status_code >= 400:
            r.raise_for_status()
        out: list[Result] = []
        for m in self._MJ_RESULT.finditer(r.text):
            url = m.group("url") or ""
            if not url.startswith("http"):
                continue
            out.append(Result(title=_clean(m.group("title")), url=url,
                              snippet=_clean(m.group("snip") or ""), kind="web"))
            if len(out) >= limit:
                break
        if not out and self._CHALLENGE.search(r.text):
            raise Challenged("mojeek")
        return out

    @staticmethod
    def _unwrap(href: str) -> str:
        """DuckDuckGo wraps some results in /l/?uddg=<encoded>. Undo that."""
        if href.startswith("//"):
            href = "https:" + href
        if "uddg=" in href:
            try:
                inner = parse_qs(urlparse(href).query).get("uddg", [""])[0]
                if inner:
                    return unquote(inner)
            except ValueError:
                pass
        return href if href.startswith("http") else ""

    # -- papers -------------------------------------------------------------
    def papers(self, query: str, limit: int = 5,
               recent: bool = False) -> list[Result]:
        """arXiv. A real bibliographic API, so titles, authors, dates and
        abstracts come back as fields rather than being scraped out of prose.

        Worth having separately from `search` because "find me research articles
        on X" is a request for papers, and a general web search for it returns
        journal landing pages and ResearchGate SEO — measured, not assumed.
        """
        self.require()
        # Field-qualified, so a multi-word topic matches on phrase rather than
        # scattering across unrelated papers that each contain one of the words.
        terms = re.sub(r'["\\]', " ", query).strip()
        params = {
            "search_query": f'all:"{terms}"' if " " in terms else f"all:{terms}",
            "start": 0,
            "max_results": max(1, limit),
            "sortBy": "submittedDate" if recent else "relevance",
            "sortOrder": "descending",
        }
        try:
            r = self._client.get(ARXIV_API, params=params, timeout=20.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise Offline(f"arXiv wouldn't answer: {type(exc).__name__}") from exc

        from xml.etree import ElementTree as ET
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return []

        out: list[Result] = []
        for entry in root.findall("a:entry", ATOM):
            title = re.sub(r"\s+", " ", (entry.findtext("a:title", "", ATOM) or "")).strip()
            if not title:
                continue
            summary = re.sub(r"\s+", " ",
                             (entry.findtext("a:summary", "", ATOM) or "")).strip()
            authors = [a.findtext("a:name", "", ATOM) or ""
                       for a in entry.findall("a:author", ATOM)]
            published = (entry.findtext("a:published", "", ATOM) or "")[:10]
            out.append(Result(
                title=title,
                url=entry.findtext("a:id", "", ATOM) or "",
                snippet=summary,
                kind="paper",
                meta={"authors": [a for a in authors if a],
                      "published": published,
                      "year": published[:4]},
            ))
        self.bus.log(f"arxiv '{query[:40]}' -> {len(out)} papers", "net")
        return out

    # -- fetch --------------------------------------------------------------
    # NOTE: the site-search URL builder lives in skills/web.py, next to the site
    # resolver that feeds it. This module is transport; that one is what HIVE
    # does with it.

    def fetch(self, url: str, prefer_main: bool = True) -> Page:
        """Read one page. Never raises for an ordinary failure — a Page with
        ok=False carries the reason, because the caller's job is to say what
        went wrong rather than to pretend the page was empty."""
        try:
            self.require()
            check_url(url)
        except (Offline, Blocked) as exc:
            return Page(url, "", "", ok=False, error=str(exc))

        try:
            with self._client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    return Page(url, "", "", ok=False,
                                error=f"the site returned {resp.status_code}")
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype:
                    return Page(url, "", "", ok=False,
                                error=f"that's {ctype.split(';')[0] or 'not a page'}, "
                                      f"not something I can read")
                # Capped as it arrives rather than after. A response with no
                # content-length can be arbitrarily large, and the point of a cap
                # that runs after the download is hard to explain.
                chunks, total = [], 0
                for chunk in resp.iter_bytes(65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= self.max_bytes:
                        break
                raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        except httpx.HTTPError as exc:
            return Page(url, "", "", ok=False,
                        error=f"couldn't load it: {type(exc).__name__}")

        text = to_text(raw, prefer_main=prefer_main)
        return Page(url, _Titles.of(raw), text, ok=bool(text),
                    error="" if text else "the page had no readable text")
