"""New claims about Hooly: exactly 1,000, each true, each in the forms the referee reads.

A claim is only as good as the evidence behind it and the text that states it, so
the generator holds to two rules.

- **Truth from sure evidence only.** Every value a claim holds comes from rows
  labelled `true` (the practice rows, and any other labelled rows of the input),
  never from the system's own labels. A `fact` or `compare` claim is judged on
  29 Oct 2026, so its values must still hold then: the last labelled value of the
  fact, with nothing after it to doubt it.
- **Every text read back as the referee reads it.** The text is characterized with
  the system's own reader (`characterize.py`, `values.py`): it must name the
  claim's subject, state the claim's value exactly and no other of its kind, and
  hold no negation, no date or time word of its own, as `claim-forms.md` asks. A
  claim whose text fails is dropped, not repaired.

Candidates of the five types are scored by how sure their evidence is, and the
1,000 kept are the surest, with every type represented.

    python -m hype.claims                    # -> claims.csv, from the practice data
    python -m hype.claims --out my.csv --show 20
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as Date

from .characterize import DEFAULT_LEXICON, DEFAULT_PRACTICE, AttributeIndex, StatementCharacterizer
from .knowledge import _magnitude, day_number
from .lexicon import Lexicon, normalize_name
from .resolve import read_rows
from .values import DATA_SCALE, DURATION_SECONDS, NUMBER_WORDS, same_value, singular

AS_OF = "2026-10-29"
TARGET = 1000
CLAIM_TYPES = ("fact", "compare", "change", "at_event", "relation")

# claim-forms.md, the words a sentence of a fact, at_event or relation may not hold, nor any compare
MONTHS_ETC = ("January February March April May June July August September October November December "
              "spring summer autumn winter Monday Tuesday Wednesday Thursday Friday Saturday Sunday").split()
TIME_WORDS = ["ago", "yesterday", "tomorrow", "once", "formerly", "previously", "originally", "initially",
              "recently", "lately", "soon", "eventually", "then", "before", "after", "since", "until", "till",
              "earlier", "later", "used to", "last week", "last month", "last quarter", "last year", "next week",
              "next month", "next quarter", "next year", "in the past", "in the future", "at the time", "at that time"]
NEGATION = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b", re.IGNORECASE)
COMPARE_PHRASES = {
    "amounts": {"gt": ["more", "higher", "greater", "larger", "bigger", "longer"],
                "lt": ["less", "fewer", "lower", "smaller", "shorter", "cheaper"]},
}
ACRONYMS = {"ceo", "cto", "sla", "api", "cli", "sdk", "ai", "eu", "ide", "ar", "soc2", "p99", "ad"}
# attributes whose value is a past event: once it held, it holds
SETTLED_DATES = {"launched", "released", "founded", "joined", "opened", "published", "announced", "beta_date",
                 "general_availability_date", "beta_opened", "chair_since", "soc2_audit_start", "date",
                 "generally_available"}  # not a "latest" or "stable" release: those move with each release
DATE_VERBS = {"launched": "launched", "released": "was released", "founded": "was founded", "joined": "joined",
              "opened": "opened", "published": "was published", "announced": "was announced", "date": "took place"}
CURRENCY_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}
PHYSICAL = {"B", "KB", "MB", "GB", "TB", "mm", "cm", "m", "km", "inch", "g", "kg", "mAh", "Hz", "kHz", "MHz", "GHz", "mW", "W", "kW"}
DURATION_WORDS = {"ms": ("ms", "ms"), "s": ("second", "seconds"), "min": ("minute", "minutes"), "h": ("hour", "hours"),
                  "day": ("day", "days"), "week": ("week", "weeks"), "month": ("month", "months"), "year": ("year", "years")}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


# --- writing values as value-forms.md reads them -------------------------------------------------

def number_text(n) -> str | None:
    """Digits with commas between thousands; None for a number the referee could read two ways."""
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, int):
        return f"{n:,}"
    text = f"{n:,.6f}".rstrip("0").rstrip(".")
    if re.search(r"\.\d{3}$", text):
        return None  # 1.500 is fifteen hundred in much of Europe
    return text


def plural(noun: str, n) -> str:
    if n == 1:
        return noun
    irregular = {"person": "people", "child": "children"}
    if noun in irregular:
        return irregular[noun]
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"
    return noun + "s"


def value_text(value: dict) -> str | None:
    """The value written as value-forms.md reads it, or None when it has no safe written form."""
    kind, v = value["kind"], value.get("value")
    if value.get("op", "eq") != "eq":
        return None
    if kind == "money":
        num = number_text(v)
        if num is None:
            return None
        cur = value["currency"]
        text = f"{CURRENCY_SYMBOL[cur]}{num}" if cur in CURRENCY_SYMBOL else f"{num} {cur}"
        return text + (f" per {value['per']}" if value.get("per") else "")
    if kind == "percent":
        return f"{number_text(v) if not isinstance(v, float) else f'{v:g}'}%"
    if kind == "quantity":
        num = number_text(v)
        unit = value["unit"]
        if num is None:
            return None
        if unit in PHYSICAL:
            return f"{num} {unit}"
        if not re.fullmatch(r"[a-z]+", unit):
            return None
        return f"{num} {plural(unit, v)}"
    if kind == "duration":
        num = number_text(v)
        if num is None or value["unit"] not in DURATION_WORDS:
            return None
        one, many = DURATION_WORDS[value["unit"]]
        return f"{num} {one if v == 1 else many}"
    if kind == "date":
        parts = str(v).split("-")
        if len(parts) == 3:
            return f"{int(parts[2])} {MONTH_NAMES[int(parts[1]) - 1]} {parts[0]}"
        if len(parts) == 2:
            return f"{MONTH_NAMES[int(parts[1]) - 1]} {parts[0]}"
        return parts[0]
    if kind == "version":
        return f"version {v}"
    if kind == "name":
        return str(v)
    return None


def attribute_words(attribute: str) -> str:
    return " ".join(w.upper() if w in ACRONYMS else w for w in attribute.split("_"))


# --- the evidence -----------------------------------------------------------------------------------

@dataclass
class Observation:
    day: int
    value: dict
    true: bool          # a labelled-true row; else a labelled-false row that states it
    sole: bool          # the false row states only this fact, so this value is the false one


class Evidence:
    """The labelled observations of every fact, by day."""

    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon
        self.facts: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        self.texts: set[str] = set()

    def add(self, row: dict, facts: list[dict]) -> None:
        self.texts.add(normalize_text(row["text"]))
        label = (row.get("label") or "").strip().lower()
        if label not in ("true", "false"):
            return
        day = day_number(row["date"])
        for f in facts:
            s = self.lexicon.resolve(f["subject"])
            if s is None or f["attribute"] not in s.attributes or f["value"].get("op", "eq") != "eq":
                continue
            self.facts[(s.name, f["attribute"])].append(Observation(day, f["value"], label == "true", len(facts) == 1))

    def sorted(self, fact):
        return sorted(self.facts.get(fact, []), key=lambda o: o.day)

    def current(self, fact, as_of: int) -> tuple[dict, float] | None:
        """The value the fact holds on `as_of`, by its last labelled-true value, and how sure that is."""
        obs = [o for o in self.sorted(fact) if o.day <= as_of]
        true = [o for o in obs if o.true]
        if not true:
            return None
        last_day = true[-1].day
        last = [o for o in true if o.day == last_day]
        if any(not same_value(last[0].value, o.value) for o in last):
            return None  # two labelled values on the last day: no telling which holds
        value = last[0].value
        # the run of agreeing labelled values at the end, and anything that doubts it after it began
        run = 0
        for o in reversed(true):
            if not same_value(o.value, value):
                break
            run += 1
        if any(o.sole and same_value(o.value, value) and o.day >= last_day for o in obs if not o.true):
            return None  # a labelled-false row denies this very value later
        if fact[1] == "date" and day_number(value["value"]) > as_of:
            return None  # an event still to come may move
        settled = fact[1] in SETTLED_DATES and value["kind"] == "date" and day_number(value["value"]) <= as_of
        sure = {1: 0.80, 2: 0.88}.get(run, 0.93)
        if settled:
            sure = min(0.98, sure + 0.06)
        else:
            age = as_of - last_day
            if age > 120:
                sure *= 0.80  # an old count or price may have moved since
            elif age > 45:
                sure *= 0.92
        return value, sure

    def held_on(self, fact, day: int) -> tuple[dict, float] | None:
        """The value the fact held on `day`: stated by labelled rows of that day, or by the nearest
        labelled rows on both sides when they agree and nothing between disagrees."""
        true = [o for o in self.sorted(fact) if o.true]
        same_day = [o for o in true if o.day == day]
        if same_day:
            if all(same_value(same_day[0].value, o.value) for o in same_day):
                return same_day[0].value, 0.95 if len(same_day) > 1 else 0.9
            return None
        before = [o for o in true if o.day < day]
        after = [o for o in true if o.day > day]
        if before and after and same_value(before[-1].value, after[0].value):
            return before[-1].value, 0.88
        return None


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".!? ")


# --- reading a claim back as the referee would ----------------------------------------------------------

class ClaimChecker:
    """The rules of claim-forms.md that a text can be checked against without the universe."""

    def __init__(self, lexicon: Lexicon, characterizer: StatementCharacterizer, input_texts: set[str]):
        self.lexicon = lexicon
        self.characterizer = characterizer
        self.input_texts = input_texts

    def problems(self, kind: str, text: str, form: dict) -> list[str]:
        why = []
        if not text or len(text) > 300:
            why.append("length")
        if normalize_text(text) in self.input_texts:
            why.append("a copy of an input row")
        if re.search(r"[.!?]\s+\S", text.rstrip(".!? ")):
            why.append("more than one sentence")
        p = self.characterizer.characterize("claim", text, AS_OF)
        named = [s.subject for s in p.subjects]
        name_spans = [(s.start, s.end) for s in p.subjects] + [(e.start, e.end) for e in p.entities]
        outside = _outside(text, name_spans)
        negated = bool(NEGATION.search(outside))
        if kind == "compare":
            for side in ("left", "right"):
                if self.lexicon.resolve(form[side]["subject"]).name not in named:
                    why.append(f"{side} subject not named")
            if len(named) >= 2 and named[0] != self.lexicon.resolve(form["left"]["subject"]).name:
                why.append("left subject not first")
            if p.values:
                why.append("a value in a compare")
            if re.search(r"\d", outside) or any(re.search(rf"\b{re.escape(w)}\b", outside, re.I) for w in NUMBER_WORDS if " " not in w):
                why.append("a number in a compare")
            if negated:
                why.append("a negation")
            return why
        subject = self.lexicon.resolve(form["subject"]).name
        if subject not in named:
            why.append("subject not named")
        if negated:
            why.append("a negation")
        if kind == "change":
            # "from X to Y" is the change phrase itself, so its two ends are not one range (claim-forms.md, Change)
            for side in ("from", "to"):
                if not any(not v.negated and not v.approximate and v.op == "eq" and same_value(_plain(v.value), form[side])
                           for v in p.values) \
                        and not (form[side]["kind"] == "name" and normalize_name(form[side]["value"]) in normalize_name(text)):
                    why.append(f"'{side}' value not read")
            return why
        value = form["value"]
        if value["kind"] == "name":
            if normalize_name(str(value["value"])) not in normalize_name(outside_subject(text, p, subject)):
                why.append("name value not written")
        elif value["kind"] != "boolean":
            same_kind = [v for v in p.values if v.kind == value["kind"]]
            if not any(v.stated and v.op == "eq" and same_value(_plain(v.value), value) for v in same_kind):
                why.append("value not read")
            if len(same_kind) > 1:
                why.append("another value of its kind")
        if value["kind"] != "date" and any(v.kind == "date" for v in p.values):
            why.append("a date of its own")
        allowed = " ".join(text[s:e] for s, e in name_spans)
        for w in MONTHS_ETC:
            if re.search(rf"\b{w}\b", outside) and value["kind"] != "date":
                why.append(f"the word {w}")
        for w in TIME_WORDS:
            if re.search(rf"\b{re.escape(w)}\b", outside, re.I) and not (kind == "at_event" and w == "at the time"):
                why.append(f"the time word '{w}'")
        return why


def _outside(text: str, spans: list[tuple[int, int]]) -> str:
    out = list(text)
    for s, e in spans:
        for i in range(s, e):
            out[i] = " "
    return "".join(out)


def outside_subject(text: str, profile, subject: str) -> str:
    spans = [(s.start, s.end) for s in profile.subjects if s.subject == subject]
    return _outside(text, spans)


def _plain(value: dict) -> dict:
    return {k: v for k, v in value.items() if k != "op"}


# --- the candidates --------------------------------------------------------------------------------------

@dataclass
class Claim:
    type: str
    text: str
    form: dict
    confidence: float
    key: tuple          # what the claim is about, to keep the 1,000 varied


def fact_text(subject: str, attribute: str, value: dict) -> str | None:
    vt = value_text(value)
    if vt is None:
        return None
    words = attribute_words(attribute)
    if value["kind"] == "date" and attribute in DATE_VERBS:
        return f"{subject} {DATE_VERBS[attribute]} on {vt}." if "-" in str(value["value"]) and len(str(value["value"])) == 10 \
            else f"{subject} {DATE_VERBS[attribute]} in {vt}."
    if value["kind"] == "quantity" and singular(attribute.split("_")[-1]) == value["unit"]:
        return f"{subject} has {vt}."
    return f"The {words} of {subject} is {vt}."


def _related(fa: tuple[str, str], fb: tuple[str, str], kind: str) -> bool:
    """Whether comparing two facts says something: one attribute of two subjects (two prices), two
    attributes of one subject that share their words (inner and cover screen), or two dates, which
    always tell an order of events. An uptime against a hallucination rate is true and says nothing."""
    if kind == "date":
        return True
    if fa[1] == fb[1]:
        return True
    if fa[0] == fb[0]:
        shared = set(fa[1].split("_")) & set(fb[1].split("_")) - {"of", "per", "the"}
        return bool(shared)
    return False


def _compare_group(value: dict) -> tuple | None:
    """Values in one group compare (claim-forms.md, Compare): one kind, units that convert, prices of one
    currency and period. Names and yes-or-no values do not compare, and versions are left out as fragile."""
    kind = value["kind"]
    if kind == "money":
        return ("money", value["currency"], value.get("per"))
    if kind == "percent":
        return ("percent",)
    if kind == "quantity":
        return ("quantity", "data") if value["unit"] in DATA_SCALE else ("quantity", value["unit"])
    if kind == "duration":
        return ("duration", "clock") if value["unit"] in DURATION_SECONDS else ("duration", "calendar")
    if kind == "date":
        return ("date",)
    return None


class ClaimGenerator:
    def __init__(self, lexicon: Lexicon, evidence: Evidence, checker: ClaimChecker, as_of: str = AS_OF):
        self.lexicon = lexicon
        self.ev = evidence
        self.checker = checker
        self.as_of = day_number(as_of)
        self.rejected: dict[str, int] = defaultdict(int)

    def _keep(self, claim: Claim | None) -> Claim | None:
        if claim is None:
            return None
        problems = self.checker.problems(claim.type, claim.text, claim.form)
        if problems:
            for p in problems:
                self.rejected[f"{claim.type}: {p}"] += 1
            return None
        return claim

    def current_values(self) -> dict:
        out = {}
        for fact in self.ev.facts:
            got = self.ev.current(fact, self.as_of)
            if got:
                out[fact] = got
        return out

    def facts(self, current) -> list[Claim]:
        out = []
        for (s, a), (value, sure) in current.items():
            if value["kind"] == "boolean":
                if not value["value"]:
                    continue
                text = f"{s} has {attribute_words(a)}."
            else:
                text = fact_text(s, a, value)
            if text:
                c = self._keep(Claim("fact", text, {"subject": s, "attribute": a, "value": value}, sure, ("fact", s, a)))
                if c:
                    out.append(c)
        return out

    def compares(self, current) -> list[Claim]:
        groups: dict[tuple, list] = defaultdict(list)
        for fact, (value, sure) in current.items():
            if any(ch.isdigit() for ch in fact[1]):
                continue  # attribute words with digits would put a number in the sentence
            g = _compare_group(value)
            if g:
                groups[g].append((fact, value, sure))
        out = []
        for g, members in groups.items():
            for (fa, va, sa), (fb, vb, sb) in itertools.combinations(members, 2):
                if fa == fb or not _related(fa, fb, g[0]):
                    continue
                ma, mb = _magnitude(va), _magnitude(vb)
                if ma is None or mb is None:
                    continue
                if g[0] == "date" and (len(str(va["value"])) != 10 or len(str(vb["value"])) != 10):
                    continue  # dates compare at one precision only
                if ma == mb:
                    op = "eq"
                elif ma > mb:
                    op = "gt"
                else:
                    (fa, va, sa), (fb, vb, sb) = (fb, vb, sb), (fa, va, sa)
                    op = "gt"
                # a pair of near values is read on two current values at once: be humbler
                gap = abs(ma - mb) / max(abs(ma), abs(mb), 1e-9)
                sure = sa * sb * (1.0 if op == "eq" or gap > 0.05 else 0.9)
                for text, use_op in self._compare_texts(fa, fb, g[0], op):
                    form = {"left": {"subject": fa[0], "attribute": fa[1]}, "right": {"subject": fb[0], "attribute": fb[1]}, "op": use_op}
                    c = self._keep(Claim("compare", text, form, sure, ("compare", fa, fb)))
                    if c:
                        out.append(c)
                        break
        return out

    @staticmethod
    def _compare_texts(fa, fb, kind: str, op: str):
        """A few phrasings, most natural first; the first the checker accepts is kept."""
        (sa, aa), (sb, ab) = fa, fb
        wa, wb = attribute_words(aa), attribute_words(ab)
        if kind == "date":
            phrase = {"gt": "after", "lt": "before", "eq": "on the same day as"}[op]
            # "X launched after Y was published", "X launched after the beta date of Y": each side in its
            # own words, the phrase standing between the two names
            left = f"{sa} {DATE_VERBS[aa]}" if aa in DATE_VERBS else f"The {wa} of {sa} came"
            if ab in DATE_VERBS:
                right = sb if aa == ab else f"{sb} {DATE_VERBS[ab]}"
            else:
                right = f"the {wb} of {sb}"
            yield f"{left} {phrase} {right}.", op
            yield f"The {wa} of {sa} came {phrase} the {wb} of {sb}.", op
            return
        if op == "eq":
            if aa == ab and len(wa.split()) <= 3:
                yield f"{sa} has the same {wa} as {sb}.", op
            yield f"The {wa} of {sa} is the same as the {wb} of {sb}.", op
            return
        word = "longer" if kind == "duration" else "more" if kind == "money" else "higher"
        if aa == ab and kind == "money" and aa in ("price", "annual_price", "price_per_user"):
            yield f"{sa} costs more than {sb}.", op
        if aa == ab and len(wa.split()) <= 3:
            yield f"{sa} has a {word} {wa} than {sb}.", op
        yield f"The {wa} of {sa} is {word} than the {wb} of {sb}.", op

    def changes(self) -> list[Claim]:
        out = []
        for fact, obs in self.ev.facts.items():
            true = sorted((o for o in obs if o.true), key=lambda o: o.day)
            seen = []
            for o in true:
                if not seen or not same_value(seen[-1][1], o.value):
                    seen.append((o.day, o.value, 1))
                else:
                    d, v, n = seen[-1]
                    seen[-1] = (d, v, n + 1)
            for (d1, a, n1), (d2, b, n2) in zip(seen, seen[1:]):
                if a["kind"] == "boolean" or value_text(a) is None or value_text(b) is None:
                    continue
                if a["kind"] != b["kind"] or (a["kind"] == "money" and (a.get("currency"), a.get("per")) != (b.get("currency"), b.get("per"))):
                    continue
                sure = min(0.97, 0.80 + 0.04 * min(n1, 2) + 0.04 * min(n2, 2))
                s, attr = fact
                text = f"The {attribute_words(attr)} of {s} went from {value_text(a)} to {value_text(b)}."
                c = self._keep(Claim("change", text, {"subject": s, "attribute": attr, "from": a, "to": b}, sure, ("change", s, attr, d2)))
                if c:
                    out.append(c)
        return out

    def at_events(self) -> list[Claim]:
        out = []
        for event in self.lexicon.events:
            subj = self.lexicon.resolve(event)
            if subj is None:
                continue
            got = self.ev.current((subj.name, "date"), self.as_of)
            if not got or len(str(got[0]["value"])) != 10:
                continue
            day, event_sure = day_number(got[0]["value"]), got[1]
            for fact in self.ev.facts:
                if fact[0] == subj.name:
                    continue
                held = self.ev.held_on(fact, day)
                if not held or held[0]["kind"] in ("boolean", "date"):
                    continue
                vt = value_text(held[0])
                if vt is None:
                    continue
                s, a = fact
                text = f"At {subj.name}, the {attribute_words(a)} of {s} was {vt}."
                form = {"subject": s, "attribute": a, "event": subj.name, "value": held[0]}
                c = self._keep(Claim("at_event", text, form, held[1] * event_sure, ("at_event", s, a, subj.name)))
                if c:
                    out.append(c)
        return out

    def relations(self) -> list[Claim]:
        out = []
        for (s, link), obs in self.ev.facts.items():
            names = {}
            for o in obs:
                if o.true and o.value["kind"] == "name":
                    target = self.lexicon.resolve(str(o.value["value"]))
                    if target and target.name != s:
                        names[target.name] = names.get(target.name, 0) + 1
            for target, n_link in names.items():
                for attr in self.lexicon.subjects[target].attributes:
                    true = [o for o in self.ev.facts.get((target, attr), []) if o.true]
                    if not true or not all(same_value(true[0].value, o.value) for o in true):
                        continue  # one value only, so it held while the link named the target
                    value = true[0].value
                    if value["kind"] == "boolean":
                        continue
                    vt = value_text(value)
                    if vt is None:
                        continue
                    lw = attribute_words(link)
                    if value["kind"] == "date" and attr in DATE_VERBS:
                        text = f"The {lw} of {s} {DATE_VERBS[attr]} on {vt}." if len(str(value["value"])) == 10 else None
                    else:
                        text = f"The {attribute_words(attr)} of the {lw} of {s} is {vt}."
                    if not text:
                        continue
                    sure = min(0.95, 0.82 + 0.03 * min(n_link, 2) + 0.03 * min(len(true), 2))
                    form = {"subject": s, "link": link, "target": target, "attribute": attr, "value": value}
                    c = self._keep(Claim("relation", text, form, sure, ("relation", s, link, attr)))
                    if c:
                        out.append(c)
        return out

    def generate(self, target: int = TARGET) -> list[Claim]:
        current = self.current_values()
        pools = {"fact": self.facts(current), "compare": self.compares(current), "change": self.changes(),
                 "at_event": self.at_events(), "relation": self.relations()}
        for pool in pools.values():
            pool.sort(key=lambda c: -c.confidence)
        # each type up to its share first, the surest of each, then the surest of all for what is left;
        # the claims that connect two facts (compare, change, at_event, relation) take most of the 1,000
        share = {"relation": 0.06, "compare": 0.28, "change": 0.24, "at_event": 0.18, "fact": 0.24}
        chosen, used = [], set()
        uses: dict = defaultdict(int)

        def take(c: Claim) -> bool:
            if c.type == "compare":
                fa, fb = c.key[1], c.key[2]
                if uses[fa] >= 6 or uses[fb] >= 6:
                    return False  # no fact in more than a few pairs, so the compares stay varied
                uses[fa] += 1
                uses[fb] += 1
            chosen.append(c)
            used.add(id(c))
            return True

        for kind in ("relation", "compare", "change", "at_event", "fact"):
            taken = 0
            for c in pools[kind]:
                if taken >= round(share[kind] * target):
                    break
                taken += take(c)
        rest = sorted((c for kind in CLAIM_TYPES for c in pools[kind] if id(c) not in used), key=lambda c: -c.confidence)
        for c in rest:
            if len(chosen) >= target:
                break
            take(c)
        self.pool_sizes = {k: len(v) for k, v in pools.items()}
        return sorted(chosen[:target], key=lambda c: (CLAIM_TYPES.index(c.type), -c.confidence))


# --- the table ----------------------------------------------------------------------------------------------

def claim_rows(claims: list[Claim]) -> list[dict]:
    return [{"id": f"claim-{n:04d}", "type": c.type, "text": c.text,
             "form": json.dumps(c.form, ensure_ascii=False, separators=(",", ":")),
             "label": "true", "confidence": f"{min(1.0, max(0.5, c.confidence)):.4f}"} for n, c in enumerate(claims, 1)]


def evidence_from(lexicon: Lexicon, rows: list[dict], practice: list[dict]) -> Evidence:
    """The labelled rows: practice rows with their gold forms; other labelled input rows need a form, and
    without one they only count as text a claim may not copy."""
    ev = Evidence(lexicon)
    gold = {(r["id"], normalize_text(r["text"])): json.loads(r["form"]) for r in practice}
    seen = set()
    for r in rows:
        key = (r["id"], normalize_text(r["text"]))
        form = gold.get(key) or (json.loads(r["form"]) if r.get("form") else [])
        ev.add(r, form)
        seen.add(key)
    for r in practice:
        key = (r["id"], normalize_text(r["text"]))
        if key not in seen:
            ev.add(r, json.loads(r["form"]))
    return ev


def generate_claims(lexicon_path: str = DEFAULT_LEXICON, rows: list[dict] | None = None,
                    target: int = TARGET) -> tuple[list[Claim], ClaimGenerator]:
    lexicon = Lexicon.load(lexicon_path)
    practice = read_rows(DEFAULT_PRACTICE)
    ev = evidence_from(lexicon, rows if rows is not None else practice, practice)
    characterizer = StatementCharacterizer(lexicon, AttributeIndex.from_rows(lexicon, practice))
    gen = ClaimGenerator(lexicon, ev, ClaimChecker(lexicon, characterizer, ev.texts))
    return gen.generate(target), gen


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate the 1,000 new true claims.")
    parser.add_argument("--out", default="claims.csv")
    parser.add_argument("--show", type=int, default=0, help="print this many claims of each type")
    parser.add_argument("--rejected", action="store_true", help="print why candidate texts were dropped")
    args = parser.parse_args(argv)
    claims, gen = generate_claims()
    rows = claim_rows(claims)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "type", "text", "form", "label", "confidence"])
        w.writeheader()
        w.writerows(rows)
    by_type = {k: sum(1 for c in claims if c.type == k) for k in CLAIM_TYPES}
    print(f"{len(claims)} claims: {by_type}; candidates per type: {gen.pool_sizes}")
    print(f"mean confidence {sum(c.confidence for c in claims) / max(1, len(claims)):.3f}")
    if args.rejected:
        for k, n in sorted(gen.rejected.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  dropped {n:5}  {k}")
    if args.show:
        for kind in CLAIM_TYPES:
            for c in [c for c in claims if c.type == kind][: args.show]:
                print(f"  [{kind}] {c.confidence:.2f}  {c.text}")
    print(f"wrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
