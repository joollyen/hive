"""What HIVE does with a network, phrased for the ear.

`hive/net.py` is the plumbing — connectivity, search providers, page fetch,
address safety. This module is the half that decides what is worth saying out
loud about what came back, and it is a separate file because those are genuinely
different problems: one is bounded by HTTP, the other by the fact that everything
here is going through a speaker.

Three rules govern every spoken line below, and all three come from HIVE having
been a voice-only assistant before it had a network:

**URLs are never spoken.** A read-out URL is unintelligible, takes longer than
the fact it points at, and cannot be written down at conversational speed. So
what is spoken is the title, the source, the year, the price — the things that
identify a result to a person — and the links go to the terminal pane and into
the task's markdown log, where a link is actually clickable.

**Counts before contents.** "Six results" first, then the results. A listener
cannot see how long a list is and has no way to decide whether to keep
listening; a reader can. Saying the count first is the spoken equivalent of the
scrollbar.

**A failure is a sentence, not an empty list.** "I couldn't reach the network",
"the site refused an automated request" and "there's nothing out there on that"
are three different things, and collapsing them into "no results" is the exact
class of quiet dishonesty the objective hemisphere exists to prevent.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

from ..net import Offline, Result, Web

# Prices as retailers write them, in the messy forms that survive a search
# snippet: "$129.99", "$1,299", "Your price for this item is $89.99".
_PRICE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")

# Snippet boilerplate that carries no information and eats the speaking budget.
_NOISE = re.compile(
    r"\b(?:shop|browse|find|discover|buy|order|get)\s+(?:the\s+)?"
    r"(?:best|top|great|latest|new)?\s*"
    r"(?:deals?|prices?|selection|online)\b[^.]*\.?|"
    r"\bfree (?:shipping|delivery|returns)\b[^.]*\.?|"
    r"\bprice match guarantee\b[^.]*\.?", re.I)


def _tidy(text: str, limit: int = 190) -> str:
    """A snippet, trimmed to something speakable and cut at a word boundary."""
    text = _NOISE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .·|-")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _first_sentence(text: str, limit: int = 200) -> str:
    """The opening claim of an abstract — usually the whole point of the paper."""
    body = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", body)
    out = parts[0] if parts else ""
    # Abstracts frequently open with one long scene-setting sentence and put the
    # actual contribution in the second. Two is the useful unit; three is a
    # paragraph read aloud.
    if len(out) < 90 and len(parts) > 1:
        out = f"{out} {parts[1]}"
    return _tidy(out, limit)


def _ordinal_run(items: list[str]) -> str:
    """Join spoken list items with ordinals rather than numerals.

    prosody.py rewrites a "1." enumerator into "First," precisely because Kokoro
    phonemises a numeral as a word. Building the list this way to begin with
    means the rewrite has nothing to do and the phrasing is chosen here, where
    the content is known, rather than recovered downstream.

    Every item is terminated. An abstract cut mid-flight ends without
    punctuation, and without a full stop the next ordinal runs straight on —
    heard on a real run as "…gained much attention recently Beyond that, First,"
    which is two list items, a section break and a sentence boundary all
    delivered as one continuous clause.
    """
    words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth",
             "Seventh", "Eighth", "Ninth", "Tenth"]
    out = []
    for i, item in enumerate(items):
        body = item.strip()
        if body and body[-1] not in ".!?…":
            body += "."
        out.append(f"{words[i] if i < len(words) else 'Then'}, {body}")
    return " ".join(out)


def _count(n: int, singular: str, plural: str = "") -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


# --- research ---------------------------------------------------------------

def research(web: Web, topic: str, limit: int = 4) -> tuple[str, list[Result]]:
    """Papers on a subject, from arXiv first and the open web second.

    arXiv is queried first and preferred because "find me research articles on
    neural networks" is a request for papers, and a general web search for that
    phrase returns journal landing pages, ResearchGate topic hubs and SEO farms —
    measured on this box against exactly that query. arXiv returns titles,
    authors, dates and abstracts as structured fields, so what HIVE says about a
    paper is read off the record rather than inferred from a search snippet.

    The web pass still runs, because arXiv is physics, maths and computer
    science and a great many subjects are none of those. When arXiv returns
    nothing the web results carry the answer on their own, and HIVE says which
    of the two it is relying on.
    """
    papers: list[Result] = []
    web_hits: list[Result] = []

    try:
        papers = web.papers(topic, limit=limit)
    except Offline as exc:
        arxiv_trouble = str(exc)
        papers = []
    except Exception as exc:                            # provider-side breakage
        arxiv_trouble = f"arXiv didn't answer ({type(exc).__name__})"
        papers = []
    else:
        arxiv_trouble = ""

    try:
        web_hits = web.search(f"{topic} research paper OR study", limit=limit)
        search_trouble = ""
    except Offline as exc:
        search_trouble = str(exc)
    except Exception as exc:
        search_trouble = f"the search broke ({type(exc).__name__})"

    # Reported only when it cost something. If arXiv answered, a blocked web
    # search is a footnote; if neither answered, it is the whole message.
    trouble = "; ".join(t for t in (arxiv_trouble, search_trouble) if t)

    if not papers and not web_hits:
        # "Nothing exists on this" and "I was not allowed to look" are different
        # sentences and only one of them is ever true at a time.
        if trouble:
            return (f"I couldn't run that search — {trouble}. So I've got "
                    f"nothing on {topic}, and that's my problem rather than a "
                    f"result."), []
        return (f"Nothing came back on {topic}. Either it's phrased too "
                f"narrowly or it isn't indexed where I can see it."), []

    spoken: list[str] = []
    if papers:
        lines = []
        for p in papers:
            year = p.meta.get("year") or ""
            authors = p.meta.get("authors") or []
            who = authors[0].split()[-1] if authors else ""
            if who and len(authors) > 1:
                who = f"{who} and others"
            attrib = ", ".join(x for x in (who, year) if x)
            gist = _first_sentence(p.snippet, 170)
            lines.append(f"{p.title}{f' — {attrib}' if attrib else ''}. {gist}")
        spoken.append(f"{_count(len(papers), 'paper')} on arXiv. "
                      + _ordinal_run(lines))

    if web_hits:
        named = [f"{h.title} on {h.host}" for h in web_hits[:3] if h.title]
        if named:
            if papers:
                # Prose, not a second enumerated list. "Beyond that, First, …"
                # was the first thing heard on a real run, and stacking two
                # ordinal runs in one answer makes the shorter one sound like a
                # continuation of the longer.
                joined = (named[0] if len(named) == 1
                          else " and ".join([", ".join(named[:-1]), named[-1]]))
                spoken.append(f"Off arXiv there's also {joined}.")
            else:
                spoken.append("Nothing on arXiv, but on the open web, "
                              + _ordinal_run(named))

    spoken.append("Links are in my terminal and they'll be in the task note.")
    if trouble:
        spoken.append(f"One caveat: {trouble}.")
    return " ".join(spoken), papers + web_hits


# --- general look-up --------------------------------------------------------

def look_up(web: Web, llm, query: str, limit: int = 5,
            read_top: bool = True) -> tuple[str, list[Result]]:
    """Search the web and answer from what came back.

    Two stages, and the second one is what makes this different from reading a
    results page aloud. The top result is fetched and its text handed to the
    local model as grounded context, with an instruction to answer only from it.
    That keeps the division of labour the whole architecture is built on: the
    network supplies the facts, the model supplies the sentence, and the model is
    never the source of the fact.

    If the fetch fails — and it will, regularly; a great many sites refuse an
    unattended client — the snippets are spoken instead and HIVE says that is
    what it is doing. A summary invented from snippets and presented as a reading
    of the page would be the failure this whole layer was built to avoid.
    """
    try:
        hits = web.search(query, limit=limit)
    except Offline as exc:
        return f"I couldn't run that search — {exc}.", []

    if not hits:
        return f"The search came back empty for {query}.", []

    page = None
    if read_top:
        page = web.fetch(hits[0].url)

    if page is not None and page.ok and llm is not None and llm.available():
        try:
            answer = llm.complete([
                {"role": "system", "content":
                    "You answer from the supplied page text and nothing else. "
                    "You are being read aloud: prose only, no markdown, no "
                    "bullets, no URLs, no headings. If the page does not answer "
                    "the question, say so plainly in one sentence. Never pad, "
                    "never add a preamble."},
                {"role": "user", "content":
                    f"Question: {query}\n\n"
                    f"Page: {page.title}\n\n{page.text[:7000]}"},
            ], heavy=True, num_predict=420).strip()
        except Exception:
            answer = ""
        if answer:
            others = _count(len(hits) - 1, "other result")
            tail = (f" I've got {others} in the terminal." if len(hits) > 1 else "")
            return (f"From {hits[0].host}: {answer}{tail}"), hits

    # No readable page. Say what is actually known — the snippets — and be
    # explicit that they are snippets.
    reason = (page.error if page is not None and not page.ok
              else "I couldn't get a readable page out of the top result")
    listed = [f"{h.title} on {h.host}. {_tidy(h.snippet, 130)}".strip()
              for h in hits[:3] if h.title]
    return (f"{_count(len(hits), 'result')}, but {reason}, so this is the search "
            f"summaries rather than the pages themselves. "
            + _ordinal_run(listed)
            + " Links are in my terminal."), hits


# --- shopping ---------------------------------------------------------------

def product_options(web: Web, product: str, site_label: str = "",
                    site_domain: str = "", limit: int = 6) -> tuple[str, list[Result]]:
    """What a retailer has, for a product described in words.

    Searched with a `site:` operator rather than by fetching the retailer's own
    search page, and that is a deliberate choice rather than a shortcut. Every
    large retailer refuses an unattended client — Best Buy answers a plain fetch
    with a bot-protection interstitial, not a product list — so scraping the
    search page directly produces a confident summary of a CAPTCHA. The search
    index has already done the crossing, its result titles *are* the product
    names, and its snippets frequently carry the price.

    So HIVE reports what the index knows and says where the number came from,
    and the retailer's own search page is opened alongside it for the half that
    genuinely needs a browser: stock, delivery and the actual purchase.
    """
    where = f"site:{site_domain} " if site_domain else ""
    # Offline/Challenged deliberately propagates. Whether the reading worked is
    # a pass/fail question about the *task*, and only the caller knows that —
    # swallowing it here would return a polite sentence that reads like a
    # result, and the task would be filed as done with nothing in it.
    hits = web.search(f"{where}{product}", limit=limit)

    if site_domain:
        hits = [h for h in hits if site_domain in h.url] or hits

    if not hits:
        return (f"Nothing came back for {product}"
                f"{f' on {site_label}' if site_label else ''}."), []

    lines: list[str] = []
    priced = 0
    for h in hits[:4]:
        # Retailer titles run long and end in boilerplate: "CORSAIR - K70 CORE
        # RGB Mechanical Gaming Keyboard - Gray - Front_Zoom". Cut at the first
        # separator that looks like the end of the name.
        name = re.split(r"\s+[-–|]\s+(?:Front|Best Buy|Walmart|Amazon)\b",
                        h.title)[0].strip()
        price = _PRICE.search(f"{h.title} {h.snippet}")
        if price:
            priced += 1
            lines.append(f"{name}, {price.group(0)}")
        else:
            lines.append(name)

    spoken = [f"{_count(len(hits), 'listing')} for {product}"
              f"{f' on {site_label}' if site_label else ''}. "
              + _ordinal_run(lines) + "."]
    if priced:
        spoken.append("Those prices come off the search listings, so treat them "
                      "as roughly right rather than current.")
    else:
        spoken.append("No prices in the listings, so you'll want the page itself "
                      "for those.")
    spoken.append("I'm opening the search so you can see stock and delivery.")
    return " ".join(spoken), hits


# --- one page ---------------------------------------------------------------

def summarise_page(web: Web, llm, url: str,
                   question: str = "") -> tuple[bool, str]:
    """Read one named page and say what is on it."""
    page = web.fetch(url)
    if not page.ok:
        return False, f"I couldn't read that page — {page.error}."
    if llm is None or not llm.available():
        return True, (f"{page.title}. I've got the text but the model isn't "
                      f"reachable to summarise it, so it's in my terminal raw.")
    try:
        body = llm.complete([
            {"role": "system", "content":
                "You summarise a web page for someone listening, not reading. "
                "Prose only — no markdown, no bullets, no headings, no URLs. "
                "Lead with what the page actually says. Use only the supplied "
                "text; if it is truncated or unreadable, say so."},
            {"role": "user", "content":
                (f"{question}\n\n" if question else "")
                + f"Page title: {page.title}\n\n{page.text[:9000]}"},
        ], heavy=True, num_predict=600).strip()
    except Exception as exc:
        return False, f"I fetched it but the summary failed: {type(exc).__name__}."
    return True, f"{page.title}. {body}" if body else (
        False, "I fetched the page and got nothing usable out of it.")


# --- site search templates --------------------------------------------------

def site_search_url(template: str, query: str) -> str:
    """Fill a site's search template from config/apps.yaml.

    `{q}` is URL-encoded, `{raw}` is not. This is the mechanism behind "open
    YouTube and type X into the search bar": the end state is identical to
    having typed it, and it is a URL rather than an automation stack, so it
    cannot be broken by the page changing its markup.
    """
    return template.replace("{q}", quote_plus(query)).replace("{raw}", query)


def resolve_site(cfg, spoken: str) -> tuple[str, dict] | None:
    """Match a spoken site name against config/apps.yaml `search_sites`.

    Longest alias first, so "best buy" is not shadowed by a shorter alias that
    happens to be a substring of it.
    """
    sites = (cfg.apps.get("search_sites") or {})
    t = (spoken or "").strip().lower()
    if not t:
        return None
    best: tuple[int, str, dict] | None = None
    for key, spec in sites.items():
        for alias in list(spec.get("aliases", [])) + [key]:
            a = alias.lower()
            if a == t or re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t):
                if best is None or len(a) > best[0]:
                    best = (len(a), key, spec)
    return (best[1], best[2]) if best else None
