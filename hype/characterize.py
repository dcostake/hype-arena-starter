"""Statement characterization.

A statement is more than its form. Its form holds the facts it states; around
them stand the things the sentence also carries, which the form leaves out but
later stages need:

- the subjects it names explicitly, and the references it leaves to context
  (`He`, `His group`, `the editor`, `our`), which the attributes it speaks of
  help resolve;
- the values it writes, each with how the referee would read it;
- the other entities it names (`Berlin`, `Lisbon` in p-48), with the roles the
  dataset has seen them play;
- its layout (a sentence, a headline, a `Label: value` line, a quote with a
  speaker), its time words and its hedges.

`characterize_statement` builds that profile for one row. Given the row's form,
it also aligns each fact with the text and keeps what the form does not explain
as the statement's residue.

    python -m hype.characterize p-48 p-49
    python -m hype.characterize --audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from .lexicon import Lexicon, NameMatch, find_names, name_pattern, normalize_name
from .values import NEGATIONS, NUMBER_WORDS, WORD, ValueMention, ValueReader, same_value, singular

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEXICON = os.path.join(ROOT, "data", "lexicon.json")
DEFAULT_PRACTICE = os.path.join(ROOT, "data", "practice.csv")

Fact = tuple[str, str]  # (subject, attribute)

STOPWORDS = set("""
a an the and or but of to in on at for by with from as is are was were be been being has have had
do does did will would can could should may might must it its this that these those there here
we our us you your they their them he his him she her i my me which who whom what when where how
than then so too very also just only now today all any each every more most less no not new up
out into over under per about after before since until s
""".split())

PERSONAL = {"he": "person", "him": "person", "his": "person", "she": "person", "her": "person", "hers": "person"}
NEUTER = {"it": "thing", "its": "thing", "they": "group", "them": "group", "their": "group"}
FIRST_PERSON = {"we": "speaker", "us": "speaker", "our": "speaker", "ours": "speaker"}
DEFINITE_HEADS = (
    "plan app company editor product phone device team group crew board office event conference "
    "keynote model service browser watch programme program report survey round tier feature platform "
    "division release update suite assistant engine region network keyboard camera battery screen "
    "display workbook file tool"
).split()

TIME_WORDS = {
    "past": ["ago", "yesterday", "once", "formerly", "previously", "originally", "initially", "used to",
             "last week", "last month", "last quarter", "last year", "in the past", "at the time",
             "at that time", "earlier", "before", "until", "till", "was", "were", "had"],
    "present": ["now", "today", "currently", "still", "yet", "this year", "this quarter", "this month",
                "so far", "to date", "as of", "from today", "starting today", "recently", "lately", "since"],
    "future": ["soon", "eventually", "will", "next week", "next month", "next quarter", "next year",
               "in the future", "plans to", "planned", "upcoming", "later", "coming"],
}
MONTH_OR_SEASON = (
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"spring|summer|autumn|winter|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
)
EPISTEMIC = {
    "reported": ["says", "said", "say", "claims", "claimed", "according to", "reportedly", "told", "announced", "insists"],
    "hedged": ["might", "could", "perhaps", "maybe", "likely", "probably", "expected", "expects", "hopes",
               "aims", "estimated", "believe", "believes", "think", "thinks", "rumour", "rumor", "allegedly"],
    "emphatic": ["officially", "confirmed", "definitely", "certainly"],
}
CAPITALISED = re.compile(r"\b[A-Z][\w'’&/-]*(?:\s+(?:of\s+|de\s+|la\s+)?[A-Z][\w'’&/-]*)*")
NOT_PROPER = set("""
I A An The This That These Those Our We He She It They His Her Its Their From Today Now Before After
Since Until Release Stable Opening Staff Cycle Standard Cool Archive Availability Documents Family
Noise Price Prices Pricing Version Launch Each Every All New Note Notes Update Our Your Yes No Not
""".split())


# --- the attribute index -----------------------------------------------------

@dataclass
class FactSignature:
    """What the dataset has seen of one fact: its kinds, units and name values."""

    kinds: Counter = field(default_factory=Counter)
    units: Counter = field(default_factory=Counter)
    names: Counter = field(default_factory=Counter)
    rows: int = 0


class AttributeIndex:
    """The facts of the universe, as the lexicon and the practice forms describe them.

    It answers which facts a value could belong to (by kind and unit), which
    roles a name has played (Berlin as a city, a headquarters), and which words
    tend to speak of an attribute.
    """

    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon
        self.signatures: dict[Fact, FactSignature] = defaultdict(FactSignature)
        self.by_unit: dict[str, set[Fact]] = defaultdict(set)
        self.by_kind: dict[str, set[Fact]] = defaultdict(set)
        self.name_roles: dict[str, set[Fact]] = defaultdict(set)
        self.name_spellings: dict[str, str] = {}
        self._word_attr: dict[str, Counter] = defaultdict(Counter)
        self._attr_rows: Counter = Counter()
        self._rows = 0
        for s in lexicon.subjects.values():
            for a in s.attributes:
                self.signatures[(s.name, a)]

    @classmethod
    def build(cls, lexicon: Lexicon, practice_path: str | None = None) -> "AttributeIndex":
        rows = []
        if practice_path and os.path.exists(practice_path):
            with open(practice_path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        return cls.from_rows(lexicon, rows)

    @classmethod
    def from_rows(cls, lexicon: Lexicon, rows: list[dict]) -> "AttributeIndex":
        """An index learned from these rows only, so held-out rows stay unseen."""
        index = cls(lexicon)
        for row in rows:
            if row.get("form"):
                index.learn(row["text"], json.loads(row["form"]))
        return index

    def kinds_of(self, fact: Fact) -> set[str]:
        """The kinds a fact's value takes: as seen, else as its attribute's name suggests."""
        sig = self.signatures.get(fact)
        if sig and sig.kinds:
            return set(sig.kinds)
        return {guess_kind(fact[1])}

    def facts_of(self, subject: str) -> list[Fact]:
        s = self.lexicon.subjects.get(subject)
        return [(subject, a) for a in s.attributes] if s else []

    def learn(self, text: str, form: list[dict]) -> None:
        self._rows += 1
        name_spans = [(m.start, m.end) for m in self.lexicon.find(text)]
        words = {w for w in content_words(text, name_spans)}
        seen_attrs = set()
        for fact in form:
            subject = self.lexicon.resolve(fact["subject"])
            key = (subject.name if subject else fact["subject"], fact["attribute"])
            value = fact["value"]
            sig = self.signatures[key]
            sig.rows += 1
            sig.kinds[value["kind"]] += 1
            self.by_kind[value["kind"]].add(key)
            if "unit" in value:
                unit = singular(value["unit"]) if value["kind"] == "quantity" else value["unit"]
                sig.units[unit] += 1
                self.by_unit[unit].add(key)
            if value["kind"] == "money":
                # a monthly price and a one-off price are different facts
                sig.units[f"per:{value.get('per', 'once')}"] += 1
            if value["kind"] == "name":
                n = normalize_name(value["value"])
                sig.names[value["value"]] += 1
                self.name_roles[n].add(key)
                self.name_spellings.setdefault(n, value["value"])
            seen_attrs.add(key[1])
        for a in seen_attrs:
            self._attr_rows[a] += 1
            for w in words:
                self._word_attr[w][a] += 1

    # what the characterization asks of it

    @property
    def counted_nouns(self) -> set[str]:
        """The nouns the dataset counts: the units of its quantities, never of its durations."""
        return {u for u, facts in self.by_unit.items() if any(self.signatures[f].kinds["quantity"] for f in facts)}

    def name_patterns(self) -> list[tuple[re.Pattern, str, str, str]]:
        """Patterns for the names the dataset uses as values, longest first."""
        names = sorted(self.name_spellings.values(), key=lambda n: len(normalize_name(n)), reverse=True)
        return [(name_pattern(n), n, n, "value") for n in names]

    def compatible(self, value: dict) -> set[Fact]:
        """The facts a value could be the value of, by its kind and unit.

        A fact the practice data never shows is compatible by the kind its
        attribute's name suggests, whatever the unit.
        """
        kind = value["kind"]
        if kind == "quantity":
            seen = set(self.by_unit.get(singular(value["unit"]), ()))
            if value["unit"] in ("GB", "MB", "TB", "KB", "B"):
                for u in ("GB", "MB", "TB", "KB", "B"):
                    seen |= self.by_unit.get(u, set())
        else:
            seen = set(self.by_kind.get(kind, ()))
        return seen | self.unseen_facts(kind)

    def unseen_facts(self, kind: str) -> set[Fact]:
        if not hasattr(self, "_unseen"):
            self._unseen: dict[str, set[Fact]] = defaultdict(set)
            for fact, sig in self.signatures.items():
                if not sig.kinds:
                    self._unseen[guess_kind(fact[1])].add(fact)
        return self._unseen.get(kind, set())

    def word_weight(self, word: str, attribute: str) -> float:
        """How much a word speaks of an attribute: a smoothed log-odds."""
        hits = self._word_attr.get(word, {}).get(attribute, 0)
        if not hits:
            return 0.0
        word_rows = sum(self._word_attr[word].values())
        p_given_word = hits / (word_rows + 1)
        p_attr = (self._attr_rows[attribute] + 1) / (self._rows + 1)
        return max(0.0, math.log(p_given_word / p_attr)) * min(1.0, hits / 2)


_KIND_HINTS = [
    ("version", r"version"),
    ("money", r"price|cost|fee|rent|credit|bonus|amount|revenue|cash|bounty|guidance"),
    ("percent", r"rate$|share|percent|margin|uptime|satisfaction|growth|acceptance|accuracy|response_rate|_cut$|renewable"),
    ("date", r"date$|^launched$|^released$|^opened$|^founded$|^joined$|^announced$|^published$|_start$|_end$|^beta_opened$|since$|available$|^next_"),
    ("duration", r"time$|length$|duration|retention|window|trial|latency|delay|cadence|interval|promise$|^open_for$|battery_life|first_response"),
    ("name", r"^ceo$|^cto$|lead$|^head|chair$|owner|host|partner|provider|employer|auditor|city|headquarters|venue|region$|speaker|sponsor$|investor|run_by|^founders$"),
    ("boolean", r"^is_|^has_|^had_|_included$|certified|open_source|encrypt|offline|mode$|^fully_|^raised_|available|option$|^anonymous$|audited|^ai_|sign_on|programme$|tier$|^on_device|messaging|booking|sync$|saver$|sidebar$|cancelling$|livestreamed$|discontinued$|^replay"),
]


def guess_kind(attribute: str) -> str:
    """The kind an attribute's name suggests, for an attribute the practice data never shows."""
    for kind, pattern in _KIND_HINTS:
        if re.search(pattern, attribute):
            return kind
    return "quantity"


def content_words(text: str, masked: list[tuple[int, int]] = ()) -> list[str]:
    words = []
    for m in WORD.finditer(text):
        if any(m.start() < e and s < m.end() for s, e in masked):
            continue
        w = singular(m.group(0).lower().strip("'’-"))
        if len(w) > 1 and w not in STOPWORDS:
            words.append(w)
    return words


# --- the profile --------------------------------------------------------------

@dataclass
class Span:
    text: str
    start: int
    end: int


@dataclass
class SubjectMention(Span):
    subject: str
    via: str               # "name" or "alias"
    is_event: bool
    ambiguous: bool        # a one-word alias written in lower case, such as "outage" or "search"
    negated: bool


@dataclass
class EntityMention(Span):
    """A name that is not a lexicon subject: a city, a person, a company."""

    name: str
    status: str            # "known" (a value the dataset has used) or "unknown"
    roles: list[Fact]      # facts whose value it has been
    negated: bool


@dataclass
class Reference(Span):
    """A reference the sentence leaves to its context."""

    kind: str              # "personal", "possessive", "neuter", "group", "speaker", "definite"
    refers_to: str         # "person", "thing", "group", "speaker"
    head: str | None = None


@dataclass
class Marker(Span):
    category: str


@dataclass
class AttributeCandidate:
    subject: str
    attribute: str
    score: float
    subject_in_text: bool
    evidence: list[str]


@dataclass
class FactAlignment:
    subject: str
    attribute: str
    value: dict
    subject_mode: str                # "explicit", "alias", "contextual"
    value_span: Span | None
    value_mode: str                  # "exact", "qualified", "polarity", "missing"


@dataclass
class StatementProfile:
    id: str
    date: str
    text: str
    layout: str                      # "sentence", "headline", "key_value", "quote"
    label: str | None                # the label of a "Label: value" line
    speaker: str | None
    subjects: list[SubjectMention]
    entities: list[EntityMention]
    values: list[ValueMention]
    references: list[Reference]
    temporal: list[Marker]
    epistemic: list[Marker]
    negation: list[Span]
    attribute_candidates: list[AttributeCandidate]
    needs_context: bool
    facts: list[FactAlignment] = field(default_factory=list)
    residue: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- characterization ---------------------------------------------------------

class StatementCharacterizer:
    """Builds the profile of each statement against one lexicon and attribute index."""

    def __init__(self, lexicon: Lexicon, index: AttributeIndex, max_candidates: int = 8):
        self.lexicon = lexicon
        self.index = index
        self.reader = ValueReader(counted_nouns=index.counted_nouns)
        self.max_candidates = max_candidates
        subject_keys = {normalize_name(s) for subj in lexicon.subjects.values() for s in subj.surfaces}
        self._entity_patterns = [p for p in index.name_patterns() if normalize_name(p[1]) not in subject_keys]

    def characterize(self, row_id: str, text: str, date: str = "", form: list[dict] | None = None) -> StatementProfile:
        layout, label, speaker, body_start = self._layout(text)
        subjects = self._subjects(text)
        taken = [(s.start, s.end) for s in subjects]
        entities = self._entities(text, taken, body_start)
        masked = taken + [(e.start, e.end) for e in entities]
        values = self.reader.read(text, masked=masked)
        references = self._references(text, masked)
        temporal = self._temporal(text, masked)
        epistemic = self._epistemic(text, speaker)
        negation = [Span(m.group(0), m.start(), m.end()) for m in WORD.finditer(text)
                    if m.group(0).lower() in NEGATIONS or m.group(0).lower().replace("’", "'").endswith("n't")]
        candidates = self._attribute_candidates(text, label, subjects, values, entities, masked)
        needs_context = bool(references) and not subjects or (not subjects and bool(values))
        profile = StatementProfile(
            id=row_id, date=date, text=text, layout=layout, label=label, speaker=speaker,
            subjects=subjects, entities=entities, values=values, references=references,
            temporal=temporal, epistemic=epistemic, negation=negation,
            attribute_candidates=candidates, needs_context=needs_context,
        )
        if form is not None:
            self._align(profile, form)
        return profile

    # layout ------------------------------------------------------------------
    def _layout(self, text: str) -> tuple[str, str | None, str | None, int]:
        stripped = text.strip()
        m = re.match(r"^([^:.!?]{1,60}?):\s+(\S.*)$", stripped)
        if m:
            head = m.group(1).strip()
            if self._is_person(head):
                return "quote", None, head, m.start(2)
            return "key_value", head, None, m.start(2)
        if re.match(r"^[\"“]", stripped) or re.search(r"[\"”]\s*,?\s+(?:he|she|they|\w+)\s+(?:says|said)\b", stripped):
            return "quote", None, None, 0
        if not re.search(r"[.!?][\"”’)]*$", stripped):
            return "headline", None, None, 0
        return "sentence", None, None, 0

    def _is_person(self, name: str) -> bool:
        n = normalize_name(name)
        roles = self.index.name_roles.get(n)
        if roles and any(a in ("ceo", "cto", "chair", "head", "lead", "host", "keynote_speaker",
                               "head_of_product", "head_of_security", "head_of_workspace", "founders",
                               "executive_owner", "executive_sponsor", "owner", "reliability_lead")
                         for _, a in roles):
            return True
        subject = self.lexicon.resolve(name)
        return bool(subject and "joined" in subject.attributes)

    # names -------------------------------------------------------------------
    def _subjects(self, text: str) -> list[SubjectMention]:
        out = []
        for m in self.lexicon.find(text):
            subject = self.lexicon.subjects[m.key]
            ambiguous = m.via == "alias" and " " not in m.surface.strip() and m.text.islower()
            if ambiguous and re.search(r"(?:\d|\b(?:" + "|".join(NUMBER_WORDS) + r"))(?:\s+[a-z]+)?\s+$", text[:m.start]):
                continue  # "4,900,000,000 photos" counts photos; it does not name Hooly Photos
            out.append(SubjectMention(
                text=m.text, start=m.start, end=m.end, subject=m.key, via=m.via,
                is_event=subject.is_event, ambiguous=ambiguous, negated=_negated_at(text, m.start),
            ))
        return out

    def _entities(self, text: str, taken: list[tuple[int, int]], body_start: int) -> list[EntityMention]:
        out: list[EntityMention] = []
        spans = list(taken)
        for m in find_names(text, self._entity_patterns):
            if any(m.start < e and s < m.end for s, e in spans):
                continue
            spans.append((m.start, m.end))
            roles = sorted(self.index.name_roles.get(normalize_name(m.key), ()))
            out.append(EntityMention(m.text, m.start, m.end, m.key, "known", roles, _negated_at(text, m.start)))
        for start, end in _proper_runs(text):
            word = text[start:end]
            if any(start < e and s < end for s, e in spans):
                continue
            sentence_start = start in (0, body_start) or re.search(r"[.!?:\"“]\s*$", text[:start])
            if sentence_start and " " not in word:
                continue  # a capital at the start of a sentence says nothing
            if re.search(r"\d\s*$", text[:start]) or not re.search(r"[a-z]", word):
                continue  # "120 Hz", "SOC": a unit or an acronym, not a name
            spans.append((start, end))
            out.append(EntityMention(word, start, end, word, "unknown", [], _negated_at(text, start)))
        return sorted(out, key=lambda e: e.start)

    # references ----------------------------------------------------------------
    def _references(self, text: str, masked: list[tuple[int, int]]) -> list[Reference]:
        out = []
        for m in re.finditer(r"\b(he|him|his|she|her|hers|it|its|they|them|their|we|us|our|ours)\b", text, re.IGNORECASE):
            if _inside(m.start(), m.end(), masked):
                continue
            w = m.group(1).lower()
            if w in PERSONAL:
                kind = "possessive" if w in ("his", "her", "hers") else "personal"
                refers = "person"
            elif w in FIRST_PERSON:
                kind, refers = "speaker", "speaker"
            else:
                kind = "possessive" if w in ("its", "their") else ("neuter" if w in ("it",) else "group")
                refers = NEUTER[w]
            head = None
            if kind == "possessive":
                nxt = re.match(r"\s+([a-z][\w-]*)", text[m.end():])
                head = singular(nxt.group(1)) if nxt else None
            out.append(Reference(m.group(0), m.start(), m.end(), kind, refers, head))
        heads = "|".join(DEFINITE_HEADS)
        for m in re.finditer(rf"\b(?:the|this|that)\s+(?:\w+\s+)?({heads})s?\b", text, re.IGNORECASE):
            if _inside(m.start(), m.end(), masked):
                continue
            out.append(Reference(m.group(0), m.start(), m.end(), "definite", "thing", m.group(1).lower()))
        return sorted(out, key=lambda r: r.start)

    # time and stance -------------------------------------------------------------
    def _temporal(self, text: str, masked: list[tuple[int, int]]) -> list[Marker]:
        out = []
        for category, words in TIME_WORDS.items():
            for w in words:
                for m in re.finditer(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
                    if not _inside(m.start(), m.end(), masked):
                        out.append(Marker(m.group(0), m.start(), m.end(), category))
        for m in re.finditer(MONTH_OR_SEASON, text):
            if not _inside(m.start(), m.end(), masked) and not re.match(r"\s*\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\s+\d{4}", text[m.end():]) \
                    and not re.search(r"\d{4}\s*$|\d\s*$", text[:m.start()]):
                out.append(Marker(m.group(0), m.start(), m.end(), "month_without_year"))
        return _dedupe(out)

    def _epistemic(self, text: str, speaker: str | None) -> list[Marker]:
        out = []
        for category, words in EPISTEMIC.items():
            for w in words:
                for m in re.finditer(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
                    out.append(Marker(m.group(0), m.start(), m.end(), category))
        for m in re.finditer(r"[\"“][^\"”]+[\"”]", text):
            out.append(Marker(m.group(0), m.start(), m.end(), "quoted"))
        if speaker:
            out.append(Marker(speaker, 0, len(speaker), "speaker"))
        return _dedupe(out)

    # attributes ------------------------------------------------------------------
    def _attribute_candidates(self, text, label, subjects, values, entities, masked) -> list[AttributeCandidate]:
        named = {s.subject for s in subjects}
        words = content_words(text, masked)
        if label:
            words += content_words(label)
        scores: dict[Fact, float] = defaultdict(float)
        evidence: dict[Fact, list[str]] = defaultdict(list)

        def credit(fact: Fact, amount: float, why: str) -> None:
            scores[fact] += amount
            if why not in evidence[fact]:
                evidence[fact].append(why)

        for v in values:
            facts = self.index.compatible(v.value)
            if not facts:
                continue
            # a unit few facts share says more than one many share
            weight = 2.0 / math.sqrt(len(facts)) + (1.0 if v.kind == "quantity" else 0.0)
            for fact in facts:
                credit(fact, weight, f"{v.kind}:{v.text}")
        for e in entities:
            for fact in e.roles:
                credit(fact, 2.0 / math.sqrt(len(e.roles)), f"name:{e.name}")
        for s in subjects:
            if s.subject in named and len(subjects) > 1:
                for fact in self.index.name_roles.get(normalize_name(s.subject), ()):
                    credit(fact, 1.0, f"name:{s.subject}")
        for fact in list(scores):
            attribute = fact[1]
            attr_words = set(attribute.split("_"))
            for w in set(words):
                bonus = self.index.word_weight(w, attribute) + (1.5 if w in attr_words else 0.0)
                if bonus:
                    credit(fact, bonus * 0.5, f"word:{w}")
            if fact[0] in named:
                credit(fact, 3.0, "subject named")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: self.max_candidates]
        return [AttributeCandidate(f[0], f[1], round(s, 3), f[0] in named, evidence[f]) for f, s in ranked]

    # alignment -------------------------------------------------------------------
    def _align(self, profile: StatementProfile, form: list[dict]) -> None:
        used_values: set[int] = set()
        used_names: set[str] = set()
        for fact in form:
            subject = self.lexicon.resolve(fact["subject"])
            name = subject.name if subject else fact["subject"]
            mention = next((s for s in profile.subjects if s.subject == name), None)
            mode = "contextual" if mention is None else ("alias" if mention.via == "alias" else "explicit")
            if mention:
                used_names.add(normalize_name(name))
            value = fact["value"]
            span, value_mode = None, "missing"
            if value["kind"] == "boolean":
                value_mode = "polarity"
            elif value["kind"] == "name":
                key = normalize_name(value["value"])
                hit = next((m for m in [*profile.entities, *profile.subjects]
                            if normalize_name(getattr(m, "name", None) or m.subject) == key
                            or (self.lexicon.resolve(value["value"]) and getattr(m, "subject", None) == self.lexicon.resolve(value["value"]).name)), None)
                if hit:
                    span, value_mode = Span(hit.text, hit.start, hit.end), "exact"
                    used_names.add(normalize_name(getattr(hit, "name", None) or hit.subject))
            else:
                for i, v in enumerate(profile.values):
                    if i not in used_values and same_value(v.value, value):
                        used_values.add(i)
                        span = Span(v.text, v.start, v.end)
                        value_mode = "exact" if v.exact else "qualified"
                        break
            profile.facts.append(FactAlignment(name, fact["attribute"], value, mode, span, value_mode))
        profile.needs_context = any(f.subject_mode == "contextual" for f in profile.facts)
        profile.residue = {
            "values": [v.text for i, v in enumerate(profile.values) if i not in used_values],
            "entities": [e.name for e in profile.entities if normalize_name(e.name) not in used_names],
            "subjects": [s.subject for s in profile.subjects if normalize_name(s.subject) not in used_names],
        }


def _negated_at(text: str, pos: int) -> bool:
    from .values import _negated
    return _negated(text, pos)


def _proper_runs(text: str) -> list[tuple[int, int]]:
    """Runs of capitalised words, broken at months, weekdays and common capitals."""
    runs = []
    for m in CAPITALISED.finditer(text):
        run_start = None
        last_end = None
        for w in re.finditer(r"\S+", m.group(0)):
            token = w.group(0)
            # a common capital opens no name, but stays inside one: "Series A", not "A"
            breaks = (token in NOT_PROPER and run_start is None) or re.fullmatch(MONTH_OR_SEASON, token.rstrip(".,"))
            s, e = m.start() + w.start(), m.start() + w.end()
            if breaks or not token[0].isupper():
                if run_start is not None and not (token[0].islower() and not breaks):
                    runs.append((run_start, last_end))
                    run_start = None
                continue
            run_start = s if run_start is None else run_start
            last_end = e
        if run_start is not None:
            runs.append((run_start, last_end))
    return runs


def _inside(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)


def _dedupe(markers: list[Marker]) -> list[Marker]:
    out: list[Marker] = []
    for m in sorted(markers, key=lambda m: (m.start, -(m.end - m.start))):
        if not any(m.start >= o.start and m.end <= o.end for o in out):
            out.append(m)
    return out


_DEFAULT: StatementCharacterizer | None = None


def default_characterizer() -> StatementCharacterizer:
    global _DEFAULT
    if _DEFAULT is None:
        lexicon = Lexicon.load(DEFAULT_LEXICON)
        _DEFAULT = StatementCharacterizer(lexicon, AttributeIndex.build(lexicon, DEFAULT_PRACTICE))
    return _DEFAULT


def characterize_statement(row: dict, characterizer: StatementCharacterizer | None = None) -> StatementProfile:
    """The profile of one statement: what it names, writes and leaves to context.

    `row` holds `id`, `text` and `date`, and `form` when the row has one (as JSON
    text or as a list of facts). Without a characterizer it uses the practice
    lexicon and index.
    """
    characterizer = characterizer or default_characterizer()
    form = row.get("form")
    if isinstance(form, str):
        form = json.loads(form) if form.strip() else None
    return characterizer.characterize(row.get("id", ""), row["text"], row.get("date", ""), form)


# --- command line -------------------------------------------------------------

def _audit(rows: list[dict], characterizer: StatementCharacterizer) -> None:
    facts = Counter()
    for row in rows:
        p = characterize_statement(row, characterizer)
        for f in p.facts:
            facts["facts"] += 1
            facts[f"subject:{f.subject_mode}"] += 1
            facts[f"value:{f.value_mode}"] += 1
            if f.value_mode in ("exact", "qualified", "polarity"):
                top = [(c.subject, c.attribute) for c in p.attribute_candidates[:3]]
                facts["attribute_in_top3"] += (f.subject, f.attribute) in top
                facts["value_found"] += 1
        facts["rows_with_residue"] += any(p.residue.values())
        facts["rows_needing_context"] += p.needs_context
    total = facts["facts"]
    print(f"{len(rows)} rows, {total} facts")
    for k, v in sorted(facts.items()):
        if k != "facts":
            print(f"  {k:24} {v:5}  {v / (len(rows) if k.startswith('rows') else total):6.1%}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Characterize the statements of the practice dataset.")
    parser.add_argument("ids", nargs="*", help="row ids to print, such as p-48")
    parser.add_argument("--audit", action="store_true", help="measure how well the profiles explain the forms")
    parser.add_argument("--no-form", action="store_true", help="characterize without the rows' forms")
    args = parser.parse_args(argv)
    characterizer = default_characterizer()
    with open(DEFAULT_PRACTICE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.audit:
        _audit(rows, characterizer)
        return 0
    by_id = {r["id"]: r for r in rows}
    for rid in args.ids:
        row = dict(by_id[rid])
        if args.no_form:
            row.pop("form")
        print(json.dumps(characterize_statement(row, characterizer).to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
