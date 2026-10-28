"""The whole system as one call: rows in, verdicts out, each with the reasons behind it.

    from hype.pipeline import HypePipeline
    pipeline = HypePipeline.load()
    for v in pipeline.label([{"id": "s-1", "date": "2026-04-01", "text": "..."}]):
        print(v.label, v.confidence)
        print(v.explain())

The stages it calls, in order:

1. `segment_pages` (resolve.py): consecutive rows of one date form a page.
2. `StatementResolver.resolve_page` (resolve.py), which calls
   `StatementCharacterizer.characterize` (characterize.py) for each row,
   then resolves its facts in two passes over the page.
3. `KnowledgeBase.add_row` (knowledge.py): the practice rows, labelled, with
   their gold forms, and the new rows with the facts the resolver read.
4. `collect_evidence` (label.py), which calls `KnowledgeBase.evidence` for
   each fact: what the other rows say of it on, before and after its date.
5. `Labeller.predict_all` (label.py): the fact model, the row model and the
   two rules found in the data.

Command line, one sentence per argument as `DATE | TEXT`, or a file of such lines:

    python -m hype.pipeline "2026-04-01 | Hooly AI Pro costs €59 per month."
    python -m hype.pipeline --file sentences.txt
    python -m hype.pipeline --same-page "..." "..."   # read the sentences as one page
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from .characterize import DEFAULT_LEXICON, DEFAULT_PRACTICE, AttributeIndex, StatementProfile
from .knowledge import ClaimEvidence, KnowledgeBase, day_number
from .label import (
    SURE_READING, Labeller, RowEvidence, RowFacts, build_knowledge, collect_evidence, gold_facts,
)
from .lexicon import Lexicon, normalize_name
from .resolve import ResolvedFact, StatementResolver, read_rows, segment_pages

# the readings of the evidence worth naming to a person, most telling first
READINGS = {
    "labelled_same_day_contradicts": "a labelled true row of the same day says otherwise",
    "labelled_bracket_contradicts": "labelled true rows before and after agree on another value",
    "only_unsure_support": "the only row stating it is one the dataset marks unsure",
    "stale": "it held earlier, and no longer did on this date",
    "contradicted_alone": "no other row ever states this value",
    "agree_only_later": "it is only stated later: too early on this date",
    "first_then_contradicted": "the first statement of the fact, and later rows say otherwise",
    "held_then_changed": "it held before this date and changed after",
    "change_announced": "a new value, confirmed by later rows",
    "first_then_confirmed": "the first statement of the fact, confirmed by later rows",
    "labelled_same_day_agrees": "a labelled true row of the same day agrees",
    "labelled_bracket_agrees": "labelled true rows before and after agree with it",
    "confirmed": "the rows before agree, and none after contradicts",
    "no_evidence": "no other row states this fact",
}


@dataclass
class FactVerdict:
    subject: str
    attribute: str
    value: dict
    reading: float                 # how sure the resolver is that the sentence states this fact
    holds: float                   # how likely the fact held on the date, by the fact model
    readings: list[str]
    before: str | None             # the nearest other value before the date, as "date: value"
    after: str | None

    @property
    def name(self) -> str:
        return f"{self.subject}.{self.attribute} = {format_value(self.value)}"


@dataclass
class Verdict:
    id: str
    date: str
    text: str
    label: str
    confidence: float | None
    probabilities: dict[str, float]
    decided_by: str                # "model", "rule: settled false" or "rule: known unsure"
    facts: list[FactVerdict] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)   # values the sentence writes that the referee would not read
    profile: StatementProfile | None = None

    def form(self) -> list[dict]:
        """The facts read from the sentence, as the practice data writes a row's form."""
        return [{"subject": f.subject, "attribute": f.attribute, "value": f.value} for f in self.facts]

    def explain(self) -> str:
        conf = f"{self.confidence:.2f}" if self.confidence is not None else "none"
        probs = ", ".join(f"{k} {v:.2f}" for k, v in self.probabilities.items())
        lines = [f"{self.id}  {self.date}  {self.text}",
                 f"  -> {self.label.upper()}  confidence {conf}  ({self.decided_by}; model: {probs})"]
        if not self.facts:
            lines.append("  no fact read: the label rests on the row's voice and the model's prior")
        for f in self.facts:
            lines.append(f"  fact  {f.name}")
            lines.append(f"        read {f.reading:.2f}, holds {f.holds:.2f}"
                         + (f"; before {f.before}" if f.before else "") + (f"; after {f.after}" if f.after else ""))
            for r in f.readings:
                lines.append(f"        - {r}")
        for u in self.unread:
            lines.append(f"  not read: {u}")
        return "\n".join(lines)


@dataclass
class Reading:
    """What the system reads from one row: the facts it states, and what it writes besides them."""

    form: list[dict]
    residue: dict[str, list[str]]


def residue(profile: StatementProfile, facts: list[ResolvedFact]) -> dict[str, list[str]]:
    """What the sentence writes that its form does not explain, as the practice forms leave it out:
    values no fact takes, names that are neither a fact's subject nor its value (`Berlin` and
    `Lisbon` in p-48), and the references it leaves to its page (`His`). Empty parts are left out."""
    used_values = {(f.text, f.value.get("kind")) for f in facts if f.source == "value"}
    used_names = {normalize_name(f.subject) for f in facts}
    used_names |= {normalize_name(str(f.value["value"])) for f in facts if f.value.get("kind") == "name"}
    out = {
        "values": [v.text for v in profile.values if (v.text, v.kind) not in used_values],
        "entities": [e.name for e in profile.entities if normalize_name(e.name) not in used_names],
        "subjects": list(dict.fromkeys(s.subject for s in profile.subjects if normalize_name(s.subject) not in used_names)),
        "references": [r.text for r in profile.references],
    }
    return {k: v for k, v in out.items() if v}


def format_value(value: dict) -> str:
    v = value.get("value")
    op = {"gt": "over ", "ge": "at least ", "lt": "under ", "le": "up to "}.get(value.get("op", "eq"), "")
    if value["kind"] == "money":
        return f"{op}{v} {value['currency']}" + (f"/{value['per']}" if value.get("per") else "")
    if "unit" in value:
        return f"{op}{v} {value['unit']}"
    return f"{op}{v}"


class HypePipeline:
    """The resolver and labeller learned from practice, and the practice rows as labelled knowledge."""

    def __init__(self, lexicon: Lexicon, resolver: StatementResolver, labeller: Labeller, known: list[RowFacts]):
        self.lexicon = lexicon
        self.resolver = resolver
        self.labeller = labeller
        self.known = known

    @classmethod
    def load(cls, lexicon_path: str = DEFAULT_LEXICON, practice_path: str = DEFAULT_PRACTICE) -> "HypePipeline":
        lexicon = Lexicon.load(lexicon_path)
        practice = read_rows(practice_path)
        resolver = StatementResolver.load(lexicon, AttributeIndex.from_rows(lexicon, practice))
        known = []
        for n, page in enumerate(segment_pages(practice)):
            for row in page:
                facts = [f for f in gold_facts(lexicon, row) if lexicon.resolve(f[0][0])]
                known.append(RowFacts(row, f"practice-{n}", facts, None, row["label"]))
        return cls(lexicon, resolver, Labeller.load(), known)

    def is_practice(self, row: dict) -> bool:
        """Whether the row is one of the practice rows the image holds, forms included."""
        if not hasattr(self, "_practice_keys"):
            self._practice_keys = {(r.row["id"], " ".join(r.row["text"].split())) for r in self.known}
        return (row.get("id", ""), " ".join(row.get("text", "").split())) in self._practice_keys

    def label(self, rows: list[dict], same_page: bool = True) -> list[Verdict]:
        """Verdicts for the rows whose `label` is empty; see `analyse`."""
        return self.analyse(rows, same_page)[0]

    def analyse(self, rows: list[dict], same_page: bool = True) -> tuple[list[Verdict], dict[str, "Reading"]]:
        """Verdicts for the rows whose `label` is empty, among rows of `id`, `date`, `text` and `label`,
        and what the system reads from every row, by id: its form and its residue.

        The rows that carry a label are knowledge: the practice rows the image holds already
        (with their gold forms), the others with the facts the resolver reads from them.
        With `same_page`, consecutive rows of one date are read as one page, as in the
        input; without it, each row stands alone.
        """
        pages = segment_pages(rows) if same_page else [[r] for r in rows]
        new: list[RowFacts] = []
        labelled: list[RowFacts] = []
        readings: dict[str, Reading] = {}
        for n, page in enumerate(pages):
            for row, (profile, facts) in zip(page, self.resolver.resolve_page(page)):
                readings[row["id"]] = Reading([f.as_form() for f in facts], residue(profile, facts))
                label = (row.get("label") or "").strip().lower()
                if label and self.is_practice(row):
                    continue
                read = [((f.subject, f.attribute), f.value, f.confidence) for f in facts]
                if label in ("true", "false", "unsure"):
                    labelled.append(RowFacts(row, f"input-{n}", read, profile, label))
                else:
                    new.append(RowFacts(row, f"input-{n}", read, profile, None))
        kb = build_knowledge(self.known + labelled + new)
        evidence = collect_evidence(kb, new)
        predictions = self.labeller.predict_all(evidence)
        return [self._verdict(kb, r, ev, p) for r, ev, p in zip(new, evidence, predictions)], readings

    def _verdict(self, kb: KnowledgeBase, r: RowFacts, ev: RowEvidence, prediction) -> Verdict:
        label, confidence, probs = prediction
        model_label = max(probs, key=probs.get)
        if Labeller._settled_false(ev) and label == "false":
            decided = "rule: settled false"
        elif label == "unsure" and model_label != "unsure":
            decided = "rule: known unsure"
        else:
            decided = "model"
        facts = []
        for (fact, value, reading), features in zip(r.facts, ev.facts):
            readings = [text for key, text in READINGS.items() if key in features]
            if reading < SURE_READING and any(k in features for k in ("labelled_same_day_contradicts", "labelled_bracket_contradicts")):
                readings.append(f"read with {reading:.2f}, under the {SURE_READING} a rule needs")
            before, after = self._neighbours(kb, fact, r)
            facts.append(FactVerdict(fact[0], fact[1], value, reading, self.labeller.facts.p_fact(features),
                                     readings, before, after))
        unread = []
        if r.profile:
            for v in r.profile.values:
                if not v.stated:
                    why = "negated" if v.negated else "approximate" if v.approximate else "one end of a range" if v.in_range else "hidden characters"
                    unread.append(f"{v.text!r} ({why})")
        return Verdict(r.row["id"], r.row["date"], r.row["text"], label, confidence,
                       {k: round(v, 3) for k, v in probs.items()}, decided, facts, unread, r.profile)

    @staticmethod
    def _neighbours(kb: KnowledgeBase, fact, r: RowFacts) -> tuple[str | None, str | None]:
        """The nearest labelled values on each side of the row's date, for a person to check."""
        day = day_number(r.row["date"])
        sure = [o for o in kb.values_of(fact) if o.labelled and not o.refutes and not o.unsure and o.row != r.row["id"]]
        from datetime import date as Date
        show = lambda o: f"{Date.fromordinal(o.day).isoformat()}: {format_value(o.value)}"
        before = [o for o in sure if o.day <= day]
        after = [o for o in sure if o.day > day]
        return (show(max(before, key=lambda o: o.day)) if before else None,
                show(min(after, key=lambda o: o.day)) if after else None)


def parse_line(line: str, n: int) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "|" not in line:
        raise SystemExit(f"line {n}: write it as 'YYYY-MM-DD | sentence'")
    date, text = (p.strip() for p in line.split("|", 1))
    day_number(date)  # fails early on a date that is not ISO
    return {"id": f"s-{n}", "date": date, "text": text}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Label sentences about Hooly, with the reasons.")
    parser.add_argument("sentences", nargs="*", help="'YYYY-MM-DD | sentence'")
    parser.add_argument("--file", help="a file of 'YYYY-MM-DD | sentence' lines")
    parser.add_argument("--same-page", action="store_true", help="read consecutive sentences of one date as one page")
    parser.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    args = parser.parse_args(argv)
    lines = list(args.sentences)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines += f.read().splitlines()
    rows = [r for n, line in enumerate(lines, start=1) if (r := parse_line(line, n))]
    if not rows:
        parser.print_help()
        return 1
    verdicts = HypePipeline.load().label(rows, same_page=args.same_page)
    if args.json:
        out = [{"id": v.id, "date": v.date, "text": v.text, "label": v.label, "confidence": v.confidence,
                "decided_by": v.decided_by, "probabilities": v.probabilities,
                "facts": [{"fact": f.name, "read": round(f.reading, 3), "holds": round(f.holds, 3),
                           "readings": f.readings, "before": f.before, "after": f.after} for f in v.facts],
                "not_read": v.unread} for v in verdicts]
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print("\n\n".join(v.explain() for v in verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
