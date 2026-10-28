"""Row labelling: whether each unlabelled row's sentence is true on its date.

A row is `false` when one of its values did not hold on its date, `unsure` when
the dataset gives no means to tell, and `true` otherwise. Two models decide it.

- **The fact model** says how likely each fact of the row held on its date,
  from what the knowledge base says of it (`knowledge.py`). Labels are given
  to rows, not facts, so it learns as a noisy-OR: a row is true when all its
  facts hold, P(row true) = the product of P(fact holds).
- **The row model** weighs that against the row itself (who speaks, in which
  voice, with which hedges and time words) and its page (a page whose other
  rows look false, or stand alone, tends to lie, or to be one nobody can check),
  over the three labels.

Both learn from the practice rows as the run sees them: a page's rows
unlabelled, with the facts the resolver reads from them, against the labelled
rows of the other pages.

    python -m hype.label --eval     # five folds of pages, as the run sees them
    python -m hype.label --train    # learn on every fold, save the models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

from .characterize import DEFAULT_LEXICON, DEFAULT_PRACTICE, ROOT, AttributeIndex, StatementProfile
from .knowledge import KnowledgeBase
from .lexicon import Lexicon
from .resolve import LogLinear, StatementResolver, read_rows, segment_pages

DEFAULT_MODEL = os.path.join(ROOT, "hype", "model", "labeller.json")
LABELS = ("true", "false", "unsure")
SURE_READING = 0.99          # a resolver confidence above which a fact is read surely enough for the rule
SETTLED_CONFIDENCE = 0.87    # the rule's precision on resolved practice rows (46 of 53); on gold forms, 98 of 98
FIRST_PERSON = re.compile(r"\b(I|me|my|mine|myself)\b")


@dataclass
class RowFacts:
    """A row as the labeller reads it: its facts, how they were read, and its profile."""

    row: dict
    page: str
    facts: list[tuple[tuple[str, str], dict, float]]   # (fact, value, confidence)
    profile: StatementProfile | None
    label: str | None                                   # the input's label, if it has one


@dataclass
class RowEvidence:
    """The features of one unlabelled row, before the models weigh them."""

    id: str
    page: str
    facts: list[dict[str, float]]     # one feature set per fact
    context: dict[str, float]         # the row's own voice
    first_person: bool


# --- features ----------------------------------------------------------------------------

def fact_features(kb: KnowledgeBase, r: RowFacts) -> list[dict[str, float]]:
    out = []
    for fact, value, confidence in r.facts:
        f = kb.evidence(fact, r.row["date"], value, r.row["id"], r.page).features()
        f["bias"] = 1.0
        f["confidence"] = confidence
        f["low_confidence"] = float(confidence < 0.6)
        f[f"kind:{value['kind']}"] = 1.0
        if value.get("op", "eq") != "eq":
            f["qualified"] = 1.0
        out.append(f)
    return out


def context_features(r: RowFacts) -> dict[str, float]:
    f: dict[str, float] = {}
    p = r.profile
    if p is None:
        return f
    f[f"layout:{p.layout}"] = 1.0
    if p.speaker:
        f["speaker"] = 1.0
        f[f"speaker:{p.speaker.split()[0].lower()}"] = 1.0
    for m in p.epistemic:
        f[f"stance:{m.category}"] = 1.0
    for m in p.temporal:
        f[f"time:{m.category}"] = 1.0
    if p.negation:
        f["negation"] = 1.0
    if FIRST_PERSON.search(p.text):
        f["first_person"] = 1.0
    if "!" in p.text:
        f["exclamation"] = 1.0
    if p.references and not p.subjects:
        f["only_references"] = 1.0
    return f


def collect_evidence(kb: KnowledgeBase, rows: list[RowFacts]) -> list[RowEvidence]:
    return [RowEvidence(r.row["id"], r.page, fact_features(kb, r), context_features(r),
                        bool(r.profile and FIRST_PERSON.search(r.profile.text))) for r in rows]


# --- the fact model: a noisy-OR over the facts of a row ------------------------------------

class FactModel:
    """P(fact holds) = sigmoid(w . x); a row is true when every fact holds."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.w: dict[str, float] = dict(weights or {})

    def p_fact(self, f: dict[str, float]) -> float:
        s = sum(self.w.get(k, 0.0) * v for k, v in f.items())
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, s))))

    def p_row(self, facts: list[dict[str, float]]) -> float:
        p = 1.0
        for f in facts:
            p *= self.p_fact(f)
        return p

    def fit(self, rows: list[list[dict[str, float]]], labels: list[str], epochs: int = 25,
            rate: float = 0.2, l2: float = 1e-3, seed: int = 5) -> None:
        data = [(facts, y == "true") for facts, y in zip(rows, labels) if facts and y in ("true", "false")]
        # an informative start: facts mostly hold
        self.w.setdefault("bias", 2.0)
        g2: dict[str, float] = defaultdict(lambda: 1e-8)
        rng = random.Random(seed)
        for _ in range(epochs):
            rng.shuffle(data)
            for facts, is_true in data:
                ps = [self.p_fact(f) for f in facts]
                p_row = math.prod(ps)
                grad: dict[str, float] = defaultdict(float)
                for f, p in zip(facts, ps):
                    # d log P(label) / d score of this fact
                    g = (1 - p) if is_true else -p_row * (1 - p) / max(1e-9, 1 - p_row)
                    for k, v in f.items():
                        grad[k] += g * v
                for k, g in grad.items():
                    g -= l2 * self.w.get(k, 0.0)
                    g2[k] += g * g
                    self.w[k] = self.w.get(k, 0.0) + rate * g / math.sqrt(g2[k])


# --- the row model -------------------------------------------------------------------------

def row_features(ev: RowEvidence, facts_model: FactModel, page: tuple[float, float, float] | None) -> dict[str, float]:
    f: dict[str, float] = {"bias": 1.0, **ev.context}
    if not ev.facts:
        f["no_facts"] = 1.0
    else:
        ps = [facts_model.p_fact(x) for x in ev.facts]
        p_row = math.prod(ps)
        f["log_p_true"] = max(-8.0, math.log(max(1e-9, p_row)))
        f["p_false"] = 1 - p_row
        f["min_p_fact"] = min(ps)
        f["facts"] = math.log1p(len(ev.facts))
        alone = [x for x in ev.facts if "no_evidence" in x or ("before_none" in x and "agree_after" not in x)]
        f["share_alone"] = len(alone) / len(ev.facts)
        f["any_alone"] = float(bool(alone))
        for key in ("new_value_last", "contradicted_alone", "first_then_confirmed", "first_then_contradicted",
                    "stale", "change_announced", "held_then_changed", "labelled_contradicts", "low_confidence",
                    "unsure_agree", "only_unsure_support", "unsure_same_day", "labelled_bracket_contradicts",
                    "labelled_same_day_contradicts", "labelled_bracket_agrees", "labelled_same_day_agrees"):
            if any(key in x for x in ev.facts):
                f[f"any:{key}"] = 1.0
        f["min_confidence"] = min(x.get("confidence", 1.0) for x in ev.facts)
    if page is not None:
        f["page:p_false"], f["page:alone"], f["page:first_person"] = page
    return f


def page_scores(evidence: list[RowEvidence], facts_model: FactModel) -> list[tuple[float, float, float]]:
    """For each row, the mean of its page's other rows: how false they look, how alone, in which voice."""
    by_page: dict[str, list[int]] = defaultdict(list)
    for i, ev in enumerate(evidence):
        by_page[ev.page].append(i)
    p_false = [1 - facts_model.p_row(ev.facts) if ev.facts else 0.0 for ev in evidence]
    alone = [sum(1.0 for x in ev.facts if "no_evidence" in x or ("before_none" in x and "agree_after" not in x)) / len(ev.facts)
             if ev.facts else 1.0 for ev in evidence]
    out = [(0.0, 0.0, 0.0)] * len(evidence)
    for members in by_page.values():
        for i in members:
            others = [j for j in members if j != i]
            if others:
                n = len(others)
                out[i] = (sum(p_false[j] for j in others) / n, sum(alone[j] for j in others) / n,
                          sum(evidence[j].first_person for j in others) / n)
    return out


class Labeller:
    def __init__(self, facts: FactModel | None = None, rows: LogLinear | None = None):
        self.facts = facts or FactModel()
        self.rows = rows or LogLinear()

    @staticmethod
    def _candidates(features: dict[str, float]) -> list[dict[str, float]]:
        return [{f"{label}|{k}": v for k, v in features.items()} for label in LABELS]

    def fit(self, evidence: list[RowEvidence], labels: list[str]) -> None:
        self.facts.fit([ev.facts for ev in evidence], labels)
        pages = page_scores(evidence, self.facts)
        examples = [(self._candidates(row_features(ev, self.facts, p)), LABELS.index(y))
                    for ev, p, y in zip(evidence, pages, labels)]
        self.rows.fit_ranking(examples, epochs=20, rate=0.2, l2=3e-4)

    def predict_all(self, evidence: list[RowEvidence]) -> list[tuple[str, float | None, dict[str, float]]]:
        pages = page_scores(evidence, self.facts)
        out = []
        for ev, p in zip(evidence, pages):
            probs = dict(zip(LABELS, self.rows.probabilities(self._candidates(row_features(ev, self.facts, p)))))
            label = max(probs, key=probs.get)
            # how sure the system is that its label is right, as the contract writes it: 0.5 to 1
            confidence = None if label == "unsure" else round(min(1.0, max(0.5, probs[label])), 4)
            if self._settled_false(ev):
                label, confidence = "false", max(confidence or 0.0, SETTLED_CONFIDENCE)
            elif label == "true" and self._known_unsure(ev):
                label, confidence = "unsure", None
            out.append((label, confidence, probs))
        return out

    @staticmethod
    def _settled_false(ev: RowEvidence) -> bool:
        """A fact, read surely, that the labelled rows leave no room for: contradicted by a labelled
        row of the same day, or between two labelled rows that agree on another value."""
        return any(x.get("confidence", 0) >= SURE_READING and
                   ("labelled_same_day_contradicts" in x or "labelled_bracket_contradicts" in x)
                   for x in ev.facts)

    def _known_unsure(self, ev: RowEvidence) -> bool:
        """The contract's own reading of unsure: a value the dataset marks, on the same day, as one it
        cannot check, with nothing else to tell it by, makes the row unsure, unless another fact looks false."""
        marked = [x for x in ev.facts if "only_unsure_support" in x and "unsure_same_day" in x]
        return bool(marked) and all(self.facts.p_fact(x) >= 0.5 for x in ev.facts if x not in marked)

    def save(self, path: str = DEFAULT_MODEL) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clean = lambda w: {k: round(v, 6) for k, v in w.items() if abs(v) > 1e-6}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"facts": clean(self.facts.w), "rows": clean(self.rows.w)}, f, indent=0, sort_keys=True)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL) -> "Labeller":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(FactModel(raw["facts"]), LogLinear(raw["rows"]))


# --- reading the rows ------------------------------------------------------------------------

def gold_facts(lexicon: Lexicon, row: dict) -> list[tuple[tuple[str, str], dict, float]]:
    out = []
    for f in json.loads(row["form"]):
        s = lexicon.resolve(f["subject"])
        out.append(((s.name if s else f["subject"], f["attribute"]), f["value"], 1.0))
    return out


def read_pages(resolver: StatementResolver, pages: list[list[dict]], labelled: set[int],
               page_ids: list[str], gold: bool) -> list[RowFacts]:
    """The rows of these pages: labelled pages with their gold forms, the others as the resolver reads them."""
    out = []
    for n, page in enumerate(pages):
        if n in labelled and gold:
            for row in page:
                out.append(RowFacts(row, page_ids[n], gold_facts(resolver.lexicon, row), None, row["label"]))
            continue
        for row, (profile, facts) in zip(page, resolver.resolve_page(page)):
            label = row["label"] if n in labelled else None
            out.append(RowFacts(row, page_ids[n], [((f.subject, f.attribute), f.value, f.confidence) for f in facts], profile, label))
    return out


def build_knowledge(rows: list[RowFacts]) -> KnowledgeBase:
    kb = KnowledgeBase()
    for r in rows:
        kb.add_row(r.row["id"], r.page, r.row["date"], r.facts, r.label)
    return kb


# --- evaluation as the run sees it -------------------------------------------------------------

def out_of_fold_evidence(lexicon: Lexicon, pages: list[list[dict]], folds: int = 5, seed: int = 3):
    """Each fold's pages unlabelled and read by a resolver that never saw them, against the others."""
    order = list(range(len(pages)))
    random.Random(seed).shuffle(order)
    fold_of = {p: i % folds for i, p in enumerate(order)}
    page_ids = [f"page-{n}" for n in range(len(pages))]
    evidence, labels, fold = [], [], []
    for k in range(folds):
        train = [n for n in range(len(pages)) if fold_of[n] != k]
        resolver = StatementResolver(lexicon, AttributeIndex.from_rows(lexicon, [r for n in train for r in pages[n]]))
        resolver.fit([pages[n] for n in train])
        rows = read_pages(resolver, pages, set(train), page_ids, gold=True)
        kb = build_knowledge(rows)
        hidden = [r for r in rows if r.label is None]
        evidence += collect_evidence(kb, hidden)
        labels += [r.row["label"] for r in hidden]
        fold += [k] * len(hidden)
        print(f"fold {k}: {len(hidden)} hidden rows", file=sys.stderr, flush=True)
    return evidence, labels, fold


def report(pred: list[tuple[str, float | None]], gold: list[str]) -> dict:
    confusion = Counter((g, p) for (p, _), g in zip(pred, gold))
    right = sum(1 for (p, _), g in zip(pred, gold) if p == g)
    decided = [(p, c, g) for (p, c), g in zip(pred, gold) if p != "unsure"]
    mean_conf = sum(c for _, c, _ in decided) / max(1, len(decided))
    acc_decided = sum(1 for p, _, g in decided if p == g) / max(1, len(decided))
    brier = sum(((c if p == g else 1 - c) - 1) ** 2 for p, c, g in decided) / max(1, len(decided))
    per = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        prec = tp / max(1, sum(n for (g, p), n in confusion.items() if p == label))
        rec = tp / max(1, sum(n for (g, p), n in confusion.items() if g == label))
        per[label] = {"precision": round(prec, 3), "recall": round(rec, 3)}
    return {
        "rows": len(gold), "accuracy": round(right / len(gold), 4),
        "majority_baseline": round(Counter(gold).most_common(1)[0][1] / len(gold), 4),
        "per_label": per,
        "decided": len(decided), "accuracy_when_decided": round(acc_decided, 4),
        "mean_confidence_when_decided": round(mean_conf, 4), "brier_when_decided": round(brier, 4),
        "confusion (gold->predicted)": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Label rows true, false or unsure.")
    parser.add_argument("--eval", action="store_true", help="cross-validate on folds of pages")
    parser.add_argument("--train", action="store_true", help="learn on every fold and save the models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--errors", type=int, default=0)
    args = parser.parse_args(argv)

    lexicon = Lexicon.load(DEFAULT_LEXICON)
    rows = read_rows(DEFAULT_PRACTICE)
    evidence, labels, fold = out_of_fold_evidence(lexicon, segment_pages(rows), folds=args.folds)

    if args.eval:
        pred: list = [None] * len(labels)
        for k in range(args.folds):
            labeller = Labeller()
            labeller.fit([e for e, j in zip(evidence, fold) if j != k], [y for y, j in zip(labels, fold) if j != k])
            test = [i for i, j in enumerate(fold) if j == k]
            for i, p in zip(test, labeller.predict_all([evidence[i] for i in test])):
                pred[i] = p[:2]
        print(json.dumps(report(pred, labels), indent=1))
        if args.errors:
            text = {r["id"]: r["text"] for r in rows}
            wrong = [(ev.id, p, g) for ev, p, g in zip(evidence, pred, labels) if p[0] != g]
            random.Random(1).shuffle(wrong)
            for rid, (p, c), g in wrong[: args.errors]:
                print(f"  {rid} gold={g} pred={p} {c} | {text[rid][:110]}")
    if args.train:
        labeller = Labeller()
        labeller.fit(evidence, labels)
        labeller.save()
        print(f"saved {DEFAULT_MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
