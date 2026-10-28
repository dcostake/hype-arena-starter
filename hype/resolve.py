"""Statement resolution: from a statement's profile, without its form, to its facts.

A statement often leaves its subject to context: `His group now counts 17
engineers` (p-48) states Holdan's team size on a page about Holdan. Resolution
reads each value of the profile and decides which fact, if any, it is the
value of, from three kinds of evidence:

- the subject: named in the sentence and how close to the value, named earlier
  on the page and how recently, the page's topic, a pronoun of its type;
- the attribute: the words of the sentence, the unit and kind of the value;
- the value itself: negated, approximate, one end of a range, compared.

The rows of one page share a date and stand together in the input, so a page is
a run of consecutive rows with one date.

Three log-linear models, learned from the practice forms, score the candidates:
one ranks the facts a value could belong to (or none), one the facts a name in
the sentence could be the value of (or none), and one decides which yes-or-no
facts the sentence states.

    python -m hype.resolve --eval        # learn on 80% of the pages, test on the rest
    python -m hype.resolve --train       # learn on every page, save the model
    python -m hype.resolve p-48          # resolve one row in its page
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from .characterize import (
    DEFAULT_LEXICON, DEFAULT_PRACTICE, ROOT, AttributeIndex, Fact, StatementCharacterizer,
    StatementProfile, content_words,
)
from .lexicon import Lexicon, normalize_name
from .values import same_value, singular

DEFAULT_MODEL = os.path.join(ROOT, "hype", "model", "resolver.json")
NULL = ("", "")  # the candidate "this value is the value of no fact"


# --- pages ---------------------------------------------------------------------

def segment_pages(rows: list[dict]) -> list[list[dict]]:
    """Runs of consecutive rows that share a date: the pages of the universe."""
    pages: list[list[dict]] = []
    for row in rows:
        if pages and pages[-1][-1].get("date") == row.get("date"):
            pages[-1].append(row)
        else:
            pages.append([row])
    return pages


class PageContext:
    """What the rows of a page read so far say about who the page is about."""

    def __init__(self, affinity: list[dict[str, float]] | None = None):
        self.row = 0
        # for each row, how much the page's other rows speak of each subject (the second pass)
        self.affinity_rows = affinity
        self.explicit_last: dict[str, int] = {}
        self.explicit_count: Counter = Counter()
        self.resolved_last: dict[str, int] = {}
        self.resolved_count: Counter = Counter()
        self.topic: str | None = None
        self.last_person: str | None = None

    def subjects(self) -> set[str]:
        return set(self.explicit_last) | set(self.resolved_last) | set(self.affinity)

    @property
    def affinity(self) -> dict[str, float]:
        if not self.affinity_rows or self.row >= len(self.affinity_rows):
            return {}
        return self.affinity_rows[self.row]

    def recency(self, subject: str, table: dict[str, int]) -> float:
        if subject not in table:
            return 0.0
        return 1.0 / (1 + self.row - table[subject])

    def advance(self, profile: StatementProfile, facts: list["ResolvedFact"], is_person) -> None:
        for s in profile.subjects:
            if s.ambiguous:
                continue
            self.explicit_last[s.subject] = self.row
            self.explicit_count[s.subject] += 1
            if is_person(s.subject):
                self.last_person = s.subject
        for f in facts:
            self.resolved_last[f.subject] = self.row
            self.resolved_count[f.subject] += 1
            if is_person(f.subject):
                self.last_person = f.subject
        if self.topic is None:
            first = [s.subject for s in profile.subjects if not s.ambiguous] or [f.subject for f in facts]
            if first:
                self.topic = first[0]
        self.row += 1


# --- results ---------------------------------------------------------------------

@dataclass
class ResolvedFact:
    subject: str
    attribute: str
    value: dict
    confidence: float
    source: str              # "value", "name" or "boolean"
    text: str                # the words that state the value
    subject_mode: str        # "explicit" (named in the sentence) or "context"

    def as_form(self) -> dict:
        return {"subject": self.subject, "attribute": self.attribute, "value": self.value}


# --- the models ------------------------------------------------------------------

class LogLinear:
    """A conditional log-linear model over sparse features, learned with AdaGrad."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.w: dict[str, float] = dict(weights or {})

    def score(self, features: dict[str, float]) -> float:
        w = self.w
        return sum(w.get(k, 0.0) * v for k, v in features.items())

    def probabilities(self, candidates: list[dict[str, float]]) -> list[float]:
        scores = [self.score(f) for f in candidates]
        top = max(scores)
        exp = [math.exp(s - top) for s in scores]
        total = sum(exp)
        return [e / total for e in exp]

    def fit_ranking(self, examples: list[tuple[list[dict[str, float]], int]], epochs: int = 12,
                    rate: float = 0.3, l2: float = 1e-4, seed: int = 7) -> None:
        """Each example is the candidates' features and the index of the right one."""
        g2: dict[str, float] = defaultdict(lambda: 1e-8)
        rng = random.Random(seed)
        order = list(range(len(examples)))
        for _ in range(epochs):
            rng.shuffle(order)
            for i in order:
                candidates, gold = examples[i]
                probs = self.probabilities(candidates)
                grad: dict[str, float] = defaultdict(float)
                for f, p in zip(candidates, probs):
                    for k, v in f.items():
                        grad[k] -= p * v
                for k, v in candidates[gold].items():
                    grad[k] += v
                for k, g in grad.items():
                    g -= l2 * self.w.get(k, 0.0)
                    g2[k] += g * g
                    self.w[k] = self.w.get(k, 0.0) + rate * g / math.sqrt(g2[k])

    def fit_binary(self, examples: list[tuple[dict[str, float], int]], epochs: int = 12, rate: float = 0.3,
                   l2: float = 1e-4, seed: int = 7) -> None:
        g2: dict[str, float] = defaultdict(lambda: 1e-8)
        rng = random.Random(seed)
        order = list(range(len(examples)))
        for _ in range(epochs):
            rng.shuffle(order)
            for i in order:
                features, y = examples[i]
                p = self.binary(features)
                for k, v in features.items():
                    g = (y - p) * v - l2 * self.w.get(k, 0.0)
                    g2[k] += g * g
                    self.w[k] = self.w.get(k, 0.0) + rate * g / math.sqrt(g2[k])

    def binary(self, features: dict[str, float]) -> float:
        s = self.score(features)
        return 1 / (1 + math.exp(-s)) if s > -30 else 0.0


# --- features --------------------------------------------------------------------

class RowView:
    """The parts of one profile that every candidate's features read."""

    def __init__(self, profile: StatementProfile, index: AttributeIndex):
        self.profile = profile
        masked = [(s.start, s.end) for s in profile.subjects] + [(e.start, e.end) for e in profile.entities]
        self.words = set(content_words(profile.text, masked))
        self.label_words = set(content_words(profile.label)) if profile.label else set()
        self.heads = {r.head for r in profile.references if r.head}
        self.refs = {r.refers_to for r in profile.references}
        self.mentions = [s for s in profile.subjects if not s.ambiguous] or list(profile.subjects)
        self.named = {s.subject for s in self.mentions}
        self.kind_counts = Counter(v.kind for v in profile.values)
        self.negated = bool(profile.negation)
        # the words outside full subject mentions: "Berlin keeps 44 desks" writes part of a name,
        # "Hooly Pixel 10 weighs 204 g" writes all of one and none of "Hooly Pixel 10 Pro"
        subject_spans = [(s.start, s.end) for s in profile.subjects]
        self.tokens = {m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9]+", profile.text)
                       if not any(m.start() < e and s < m.end() for s, e in subject_spans)}
        self._attr_score: dict[str, float] = {}
        self._near: dict[tuple[int, int], set[str]] = {}
        self._index = index
        self._masked = masked
        self.speaker_end = len(profile.speaker) + 1 if profile.speaker else -1

    def near_words(self, start: int, end: int, reach: int = 30) -> set[str]:
        """The content words within a few words of a span, on either side."""
        key = (start, end)
        if key not in self._near:
            text = self.profile.text
            left = text[max(0, start - reach):start]
            right = text[end:end + reach]
            self._near[key] = set(content_words(left)) | set(content_words(right))
        return self._near[key]

    def attribute_words(self, attribute: str) -> float:
        if attribute not in self._attr_score:
            self._attr_score[attribute] = sum(self._index.word_weight(w, attribute) for w in self.words)
        return self._attr_score[attribute]


def _attr_tokens(attribute: str) -> set[str]:
    return {t if not t.endswith("s") or len(t) < 4 else t[:-1] for t in attribute.split("_") if len(t) > 1}


class FeatureBuilder:
    def __init__(self, index: AttributeIndex, characterizer: StatementCharacterizer):
        self.index = index
        self.characterizer = characterizer
        self._person: dict[str, bool] = {}
        self._name_token_cache: dict[str, set[str]] = {}
        self._token_subjects: Counter = Counter()
        for subj in index.lexicon.subjects.values():
            words = set()
            for surface in subj.surfaces:
                words |= {w.lower() for w in re.findall(r"[A-Za-z0-9]+", surface)}
            for w in words:
                self._token_subjects[w] += 1
        self._kind_total: Counter = Counter()
        for fact, sig in index.signatures.items():
            for k, n in sig.kinds.items():
                self._kind_total[k] += n
        self._unit_total: Counter = Counter()
        for fact, sig in index.signatures.items():
            for u, n in sig.units.items():
                self._unit_total[u] += n

    def _name_tokens(self, subject: str, tokens: set[str]) -> tuple[float, float]:
        """How much of a subject's name the sentence writes, outside a full mention:
        `Berlin keeps 44 desks` writes half of `Hooly Berlin Office`."""
        if subject not in self._name_token_cache:
            words = set()
            for surface in self.index.lexicon.subjects[subject].surfaces:
                words |= {w.lower() for w in re.findall(r"[A-Za-z0-9]+", surface)}
            words -= {"hooly", "the", "of", "s", "and"}
            self._name_token_cache[subject] = words
        words = self._name_token_cache[subject]
        if not words:
            return 0.0, 0.0
        hit = words & tokens
        if not hit:
            return 0.0, 0.0
        rare = max(1.0 / self._token_subjects[w] for w in hit)
        return len(hit) / len(words), rare

    def is_person(self, subject: str) -> bool:
        if subject not in self._person:
            self._person[subject] = self.characterizer._is_person(subject)
        return self._person[subject]

    # the subject side, shared by every model
    def subject_features(self, view: RowView, ctx: PageContext, subject: str, anchor: int | None) -> dict[str, float]:
        f: dict[str, float] = {}
        mentions = [m for m in view.mentions if m.subject == subject]
        if mentions:
            f["s:in_sentence"] = 1.0
            if anchor is not None:
                dist = min(abs(m.start - anchor) for m in mentions)
                nearest = min(view.mentions, key=lambda m: abs(m.start - anchor))
                f["s:nearest"] = float(nearest.subject == subject)
                f["s:distance"] = min(dist, 120) / 120
                f["s:before_value"] = float(any(m.end <= anchor for m in mentions))
            f["s:alias"] = float(all(m.via == "alias" for m in mentions))
        else:
            f["s:absent"] = 1.0
            if not view.mentions:
                f["s:absent_no_mentions"] = 1.0
        f["s:explicit_recency"] = ctx.recency(subject, ctx.explicit_last)
        f["s:resolved_recency"] = ctx.recency(subject, ctx.resolved_last)
        f["s:topic"] = float(ctx.topic == subject)
        if ctx.row:
            f["s:page_share"] = (ctx.explicit_count[subject] + ctx.resolved_count[subject]) / ctx.row
        if not mentions and subject not in ctx.subjects():
            f["s:unseen_on_page"] = 1.0
        share, rare = self._name_tokens(subject, view.tokens)
        if share:
            f["s:name_tokens"] = share
            f["s:name_token_rare"] = rare
        if ctx.affinity:
            f["s:page_affinity"] = ctx.affinity.get(subject, 0.0)
            top = max(ctx.affinity, key=ctx.affinity.get)
            f["s:page_top"] = float(top == subject)
        person = self.is_person(subject)
        if "person" in view.refs:
            f["s:ref_person_match" if person else "s:ref_person_mismatch"] = 1.0
            if ctx.last_person == subject:
                f["s:last_person"] = 1.0
        if view.refs & {"thing", "group"} and not person:
            f["s:ref_thing_match"] = 1.0
        if "speaker" in view.refs and subject == "Hooly":
            f["s:speaker_hooly"] = 1.0
        if subject == "Hooly":
            f["s:hooly"] = 1.0
        return f

    def attribute_features(self, view: RowView, fact: Fact, kind: str, span: tuple[int, int] | None = None) -> dict[str, float]:
        subject, attribute = fact
        tokens = _attr_tokens(attribute)
        f = {
            "a:words": view.attribute_words(attribute),
            "a:tokens": float(len(tokens & view.words)),
            "a:all_tokens": float(bool(tokens) and tokens <= view.words),
            "a:label": float(len(tokens & view.label_words)),
            "a:head": float(bool(tokens & view.heads)),
        }
        if span is not None:
            near = view.near_words(*span)
            f["a:near_words"] = sum(self.index.word_weight(w, attribute) for w in near)
            f["a:near_tokens"] = float(len(tokens & near))
        sig = self.index.signatures.get(fact)
        if sig and sig.kinds:
            f["a:kind_prior"] = math.log((sig.kinds[kind] + 0.5) / (self._kind_total[kind] + 1))
            f["a:rows"] = math.log1p(sig.rows)
        else:
            f["a:unseen"] = 1.0
        facts_of_kind = sum(1 for a in self.index.facts_of(subject) if kind in self.index.kinds_of(a))
        f["a:subject_kind_ambiguity"] = 1.0 / max(1, facts_of_kind)
        return f

    def value_candidates(self, view: RowView, ctx: PageContext, i: int) -> tuple[list[Fact], list[dict[str, float]]]:
        v = view.profile.values[i]
        facts = sorted(self.index.compatible(v.value))
        near = view.named | ctx.subjects()
        # every compatible fact of a subject on the page, and the best few of the rest
        on_page = [f for f in facts if f[0] in near]
        rest = [f for f in facts if f[0] not in near]
        rest.sort(key=lambda f: -(view.attribute_words(f[1]) + self._unit_share(f, v.value)))
        candidates = on_page + rest[:12]
        feats = [self._null_features(view, v)]
        for fact in candidates:
            f = {}
            f.update(self.subject_features(view, ctx, fact[0], v.start))
            f.update(self.attribute_features(view, fact, v.kind, (v.start, v.end)))
            unit_share = self._unit_share(fact, v.value)
            if unit_share:
                f["a:unit_share"] = unit_share
            if view.kind_counts[v.kind] > 1:
                f["v:shared_kind"] = 1.0
            if v.kind == "date" and self.index.lexicon.subjects[fact[0]].is_event and fact[1] == "date":
                f["a:event_date"] = 1.0
            f[f"kind:{v.kind}"] = 1.0
            feats.append(f)
        return [NULL] + candidates, feats

    def _unit_share(self, fact: Fact, value: dict) -> float:
        """The share of the fact's values the dataset wrote in this value's unit or period."""
        sig = self.index.signatures.get(fact)
        if not sig or not sig.units:
            return 0.0
        if value["kind"] == "money":
            unit = f"per:{value.get('per', 'once')}"
        elif value["kind"] == "quantity":
            unit = singular(value["unit"])
        else:
            unit = value.get("unit")
        if not unit:
            return 0.0
        return sig.units.get(unit, 0) / sum(sig.units.values())

    @staticmethod
    def _null_features(view: RowView, v) -> dict[str, float]:
        f = {"null:bias": 1.0, f"null:kind:{v.kind}": 1.0}
        if v.negated:
            f["null:negated"] = 1.0
        if v.approximate:
            f["null:approximate"] = 1.0
        if v.in_range:
            f["null:range"] = 1.0
        if v.op != "eq":
            f["null:op"] = 1.0
        if not view.mentions:
            f["null:no_subject"] = 1.0
        return f

    # names as values
    def name_candidates(self, view: RowView, ctx: PageContext, mention) -> tuple[list[Fact], list[dict[str, float]]]:
        own = getattr(mention, "subject", None)
        key = normalize_name(getattr(mention, "name", None) or own)
        roles = self.index.name_roles.get(key, set())
        subjects = (view.named | ctx.subjects()) - {own}
        candidates = {r for r in roles}
        for s in subjects:
            for fact in self.index.facts_of(s):
                if "name" in self.index.kinds_of(fact):
                    candidates.add(fact)
        candidates.discard(None)
        candidates = sorted(c for c in candidates if c[0] != own)
        null = {"null:bias": 1.0, "null:name": 1.0}
        if own:
            null["null:name_is_subject"] = 1.0
            if ctx.topic == own:
                null["null:name_is_topic"] = 1.0
        if getattr(mention, "status", "") == "unknown":
            null["null:name_unknown"] = 1.0
        if mention.negated:
            null["null:negated"] = 1.0
        if mention.start < view.speaker_end:
            null["null:speaker"] = 1.0
        feats = [null]
        for fact in candidates:
            f = {"n:bias": 1.0}
            f.update(self.subject_features(view, ctx, fact[0], mention.start))
            f.update(self.attribute_features(view, fact, "name", (mention.start, mention.end)))
            f["n:known_role"] = float(fact in roles)
            if roles and fact not in roles:
                f["n:other_role"] = 1.0
            feats.append(f)
        return [NULL] + candidates, feats

    # yes or no
    def boolean_candidates(self, view: RowView, ctx: PageContext) -> tuple[list[Fact], list[dict[str, float]]]:
        subjects = set(view.named)
        if ctx.topic:
            subjects.add(ctx.topic)
        recent = sorted(ctx.subjects(), key=lambda s: -max(ctx.recency(s, ctx.explicit_last), ctx.recency(s, ctx.resolved_last)))
        subjects.update(recent[:3])
        candidates, feats = [], []
        for s in sorted(subjects):
            for fact in self.index.facts_of(s):
                if "boolean" not in self.index.kinds_of(fact):
                    continue
                f = {"b:bias": 1.0}
                f.update(self.subject_features(view, ctx, s, None))
                f.update(self.attribute_features(view, fact, "boolean"))
                f["b:values_in_sentence"] = min(3, len(view.profile.values)) / 3
                f["b:negated"] = float(view.negated)
                candidates.append(fact)
                feats.append(f)
        return candidates, feats


# --- the resolver ----------------------------------------------------------------

class StatementResolver:
    """Resolves the statements of a page in two passes.

    The first pass reads each row with what the page said before it. The second
    reads it again knowing what the page's other rows, before and after it, seem
    to be about: a page that never names Hooly Translate still speaks of its
    languages, its users and its accuracy.
    """

    def __init__(self, lexicon: Lexicon, index: AttributeIndex, passes: list[dict[str, LogLinear]] | None = None):
        self.lexicon = lexicon
        self.index = index
        self.characterizer = StatementCharacterizer(lexicon, index)
        self.features = FeatureBuilder(index, self.characterizer)
        self.passes = passes or []
        self.boolean_threshold = 0.35  # chosen on held-out pages: yes-or-no facts are rare, so the model is shy

    # learning
    def fit(self, pages: list[list[dict]]) -> dict:
        profiles = [[self.characterizer.characterize(r["id"], r["text"], r.get("date", ""),
                                                     json.loads(r["form"]) if r.get("form") else [])
                     for r in page] for page in pages]
        first, stats = self._fit_pass(profiles, None)
        self.passes = [first]
        affinities = [self._affinities(page) for page in profiles]
        second, stats2 = self._fit_pass(profiles, affinities)
        self.passes = [first, second]
        return {"first_pass": stats, "second_pass": stats2}

    def _fit_pass(self, pages: list[list[StatementProfile]], affinities) -> tuple[dict[str, "LogLinear"], dict]:
        value_ex, name_ex, bool_ex = [], [], []
        missed = Counter()
        for n, page in enumerate(pages):
            ctx = PageContext(affinities[n] if affinities else None)
            for profile in page:
                view = RowView(profile, self.index)
                gold_by_start = {}
                gold_facts = []
                for a in profile.facts:
                    gold_facts.append(ResolvedFact(a.subject, a.attribute, a.value, 1.0, "", "", ""))
                    if a.value_span is not None:
                        gold_by_start[(a.value_span.start, a.value["kind"] == "name")] = (a.subject, a.attribute)
                for i, v in enumerate(profile.values):
                    facts, feats = self.features.value_candidates(view, ctx, i)
                    gold = gold_by_start.get((v.start, False), NULL)
                    if gold in facts:
                        value_ex.append((feats, facts.index(gold)))
                    else:
                        missed["value"] += 1
                for m in self._name_mentions(profile):
                    facts, feats = self.features.name_candidates(view, ctx, m)
                    gold = gold_by_start.get((m.start, True), NULL)
                    if gold in facts:
                        name_ex.append((feats, facts.index(gold)))
                    else:
                        missed["name"] += 1
                gold_bools = {(a.subject, a.attribute) for a in profile.facts if a.value["kind"] == "boolean"}
                facts, feats = self.features.boolean_candidates(view, ctx)
                for fact, f in zip(facts, feats):
                    bool_ex.append((f, int(fact in gold_bools)))
                missed["boolean"] += len(gold_bools - set(facts))
                ctx.advance(profile, gold_facts, self.features.is_person)
        models = {"value": LogLinear(), "name": LogLinear(), "boolean": LogLinear()}
        models["value"].fit_ranking(value_ex)
        models["name"].fit_ranking(name_ex)
        models["boolean"].fit_binary(bool_ex)
        stats = {"value_examples": len(value_ex), "name_examples": len(name_ex),
                 "boolean_examples": len(bool_ex), "gold_not_among_candidates": dict(missed)}
        return models, stats

    def _affinities(self, page: list[StatementProfile]) -> list[dict[str, float]]:
        """For each row, how much the page's other rows speak of each subject, by the first pass."""
        per_row = []
        for profile, facts in self._run(page, self.passes[0], None):
            weight: dict[str, float] = defaultdict(float)
            for f in facts:
                weight[f.subject] = max(weight[f.subject], f.confidence)
            for m in profile.subjects:
                if not m.ambiguous:
                    weight[m.subject] = max(weight[m.subject], 0.5)
            per_row.append(weight)
        others = max(1, len(page) - 1)
        out = []
        for i in range(len(page)):
            total: dict[str, float] = defaultdict(float)
            for j, weight in enumerate(per_row):
                if j != i:
                    for subject, w in weight.items():
                        total[subject] += w
            out.append({subject: w / others for subject, w in total.items()})
        return out

    @staticmethod
    def _name_mentions(profile: StatementProfile):
        return [*profile.entities, *[s for s in profile.subjects if not s.ambiguous]]

    # resolution
    def resolve_page(self, page: list[dict]) -> list[tuple[StatementProfile, list[ResolvedFact]]]:
        profiles = [self.characterizer.characterize(r["id"], r["text"], r.get("date", "")) for r in page]
        affinity = self._affinities(profiles) if len(self.passes) > 1 else None
        return self._run(profiles, self.passes[-1], affinity)

    def _run(self, profiles: list[StatementProfile], models: dict[str, "LogLinear"], affinity):
        ctx = PageContext(affinity)
        out = []
        for profile in profiles:
            facts = self.resolve(profile, ctx, models)
            ctx.advance(profile, facts, self.features.is_person)
            out.append((profile, facts))
        return out

    def resolve(self, profile: StatementProfile, ctx: PageContext, models: dict[str, "LogLinear"]) -> list[ResolvedFact]:
        view = RowView(profile, self.index)
        # a slot is one thing the sentence writes: the chance it states no fact, and its candidates
        slots: list[tuple[float, list[ResolvedFact]]] = []
        for i, v in enumerate(profile.values):
            facts, feats = self.features.value_candidates(view, ctx, i)
            probs = models["value"].probabilities(feats)
            ranked = sorted(zip(facts[1:], probs[1:]), key=lambda x: -x[1])[:3]
            slots.append((probs[0], [self._fact(f, v.value, p, "value", v.text, view) for f, p in ranked]))
        for m in self._name_mentions(profile):
            facts, feats = self.features.name_candidates(view, ctx, m)
            probs = models["name"].probabilities(feats)
            value = {"kind": "name", "value": getattr(m, "name", None) or m.subject}
            ranked = sorted(zip(facts[1:], probs[1:]), key=lambda x: -x[1])[:3]
            slots.append((probs[0], [self._fact(f, value, p, "name", m.text, view) for f, p in ranked]))
        facts, feats = self.features.boolean_candidates(view, ctx)
        for fact, f in zip(facts, feats):
            p = models["boolean"].binary(f)
            value = {"kind": "boolean", "value": not view.negated}
            slots.append((self.boolean_threshold, [self._fact(fact, value, p, "boolean", "", view)]))
        return self._decode(slots)

    @staticmethod
    def _decode(slots: list[tuple[float, list[ResolvedFact]]]) -> list[ResolvedFact]:
        """Each slot states one fact at most, each fact takes one value, the surest first."""
        chosen: list[ResolvedFact] = []
        used: set[Fact] = set()
        for null_p, candidates in sorted(slots, key=lambda s: -max((c.confidence for c in s[1]), default=0)):
            for c in candidates:
                if c.confidence <= null_p:
                    break
                if (c.subject, c.attribute) in used:
                    continue
                used.add((c.subject, c.attribute))
                chosen.append(c)
                break
        return chosen

    def _fact(self, fact: Fact, value: dict, p: float, source: str, text: str, view: RowView) -> ResolvedFact:
        mode = "explicit" if fact[0] in view.named else "context"
        return ResolvedFact(fact[0], fact[1], value, p, source, text, mode)

    # persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        passes = [{k: {w: round(v, 6) for w, v in m.w.items() if abs(v) > 1e-6} for k, m in models.items()}
                  for models in self.passes]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"passes": passes, "boolean_threshold": self.boolean_threshold}, f, indent=0, sort_keys=True)

    @classmethod
    def load(cls, lexicon: Lexicon, index: AttributeIndex, path: str = DEFAULT_MODEL) -> "StatementResolver":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        resolver = cls(lexicon, index, [{k: LogLinear(w) for k, w in models.items()} for models in raw["passes"]])
        resolver.boolean_threshold = raw.get("boolean_threshold", 0.5)
        return resolver


# --- evaluation --------------------------------------------------------------------

def fact_matches(pred: ResolvedFact, gold: dict, lexicon: Lexicon) -> bool:
    s = lexicon.resolve(gold["subject"])
    return (s is not None and pred.subject == s.name and pred.attribute == gold["attribute"]
            and same_value(pred.value, gold["value"]))


def evaluate(resolver: StatementResolver, pages: list[list[dict]], show: int = 0) -> dict:
    tp = fp = fn = 0
    by_kind = defaultdict(Counter)
    by_mode = Counter()
    errors = []
    for page in pages:
        for (profile, preds), row in zip(resolver.resolve_page(page), page):
            gold = json.loads(row["form"])
            matched = set()
            for p in preds:
                hit = next((i for i, g in enumerate(gold) if i not in matched and fact_matches(p, g, resolver.lexicon)), None)
                if hit is None:
                    fp += 1
                    by_kind[p.value["kind"]]["fp"] += 1
                    errors.append((row["id"], row["text"], "extra", f"{p.subject}.{p.attribute}={p.value.get('value')}"))
                else:
                    matched.add(hit)
                    tp += 1
                    by_kind[p.value["kind"]]["tp"] += 1
                    by_mode[p.subject_mode] += 1
            for i, g in enumerate(gold):
                if i not in matched:
                    fn += 1
                    by_kind[g["value"]["kind"]]["fn"] += 1
                    errors.append((row["id"], row["text"], "missed", f"{g['subject']}.{g['attribute']}={g['value'].get('value')}"))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    report = {
        "facts": tp + fn, "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "right_by_subject_mode": dict(by_mode),
        "by_kind": {k: {**c, "f1": round(2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"]), 3)} for k, c in sorted(by_kind.items())},
    }
    if show:
        random.Random(1).shuffle(errors)
        report["errors"] = errors[:show]
    return report


# --- command line --------------------------------------------------------------------

def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_pages(pages: list[list[dict]], test_share: float = 0.2, seed: int = 13):
    order = list(range(len(pages)))
    random.Random(seed).shuffle(order)
    cut = int(len(pages) * test_share)
    test = {i for i in order[:cut]}
    return [p for i, p in enumerate(pages) if i not in test], [p for i, p in enumerate(pages) if i in test]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Resolve statements to facts.")
    parser.add_argument("ids", nargs="*", help="row ids to resolve in their page, such as p-48")
    parser.add_argument("--eval", action="store_true", help="learn on 80%% of the pages and test on the rest")
    parser.add_argument("--train", action="store_true", help="learn on every page and save the model")
    parser.add_argument("--errors", type=int, default=0, help="with --eval, print this many errors")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    lexicon = Lexicon.load(DEFAULT_LEXICON)
    rows = read_rows(DEFAULT_PRACTICE)
    pages = segment_pages(rows)

    if args.eval:
        train, test = split_pages(pages, seed=args.seed)
        resolver = StatementResolver(lexicon, AttributeIndex.from_rows(lexicon, [r for p in train for r in p]))
        stats = resolver.fit(train)
        print(json.dumps({"train_pages": len(train), "test_pages": len(test), **stats}, indent=1))
        report = evaluate(resolver, test, show=args.errors)
        errors = report.pop("errors", [])
        print(json.dumps(report, indent=1))
        for e in errors:
            print("  ", *e, sep=" | ")
        return 0

    index = AttributeIndex.from_rows(lexicon, rows)
    if args.train:
        resolver = StatementResolver(lexicon, index)
        print(json.dumps(resolver.fit(pages), indent=1))
        resolver.save(DEFAULT_MODEL)
        print(f"saved {DEFAULT_MODEL}")
        return 0

    resolver = StatementResolver.load(lexicon, index)
    by_id = {r["id"]: r for r in rows}
    for rid in args.ids:
        page = next(p for p in pages if any(r["id"] == rid for r in p))
        for profile, facts in resolver.resolve_page(page):
            if profile.id == rid:
                print(profile.text)
                for f in facts:
                    print("  ->", json.dumps(asdict(f), ensure_ascii=False))
                print("  gold", by_id[rid]["form"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
