"""The temporal knowledge base: what the universe's rows say each fact held, and when.

Each fact of the universe holds one value at a time and changes at dates: Hooly
Gemini speaks 6 languages until Hooly I/O 2026, then 12. A row's facts are
observations of those values on the row's date, and the rows disagree: some
pages lie, some repeat an old value, some announce a value too early.

The base keeps every observation with its weight:

- a row labelled `true` is a sure observation of each of its facts;
- a row labelled `false` is a sure observation that one of its facts did not
  hold, a refutation of its values, never a support;
- an unlabelled row is an observation weighted by how sure the resolver is of
  its fact, and by how often unlabelled rows are true.

A query asks what the base says of one value of one fact on one date, without
the row that states it, and answers with evidence (`ClaimEvidence`), not a
verdict: the labeller weighs the evidence.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as Date

from .values import DATA_SCALE, DURATION_SECONDS, same_value

Fact = tuple[str, str]


def day_number(iso: str) -> int:
    """Days since the epoch for an ISO date, reading a year or a month as its first day."""
    parts = [int(p) for p in iso.split("-")]
    while len(parts) < 3:
        parts.append(1)
    return Date(*parts).toordinal()


# --- values ---------------------------------------------------------------------

def _magnitude(value: dict) -> float | None:
    """A number that compares across units, for the amounts that order."""
    kind = value["kind"]
    v = value.get("value")
    if kind == "quantity":
        unit = value["unit"]
        return float(v) * DATA_SCALE.get(unit, 1)
    if kind == "duration":
        unit = value["unit"]
        if unit in DURATION_SECONDS:
            return float(v) * DURATION_SECONDS[unit]
        return float(v) * (30 * 86400 if unit == "month" else 365 * 86400)
    if kind in ("money", "percent"):
        return float(v)
    if kind == "date":
        return float(day_number(v))
    if kind == "version":
        return float(sum(int(p) / 1000 ** i for i, p in enumerate(str(v).split("."))))
    return None


def names_match(a: str, b: str) -> bool:
    """Two spellings of one name: `Gavin` is `Gavin Bellson` once one holds the other's words."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    return bool(ta and tb) and (ta <= tb or tb <= ta)


def values_agree(held: dict, claimed: dict) -> bool:
    """Whether a value a fact held makes a claimed value true, its comparison included."""
    if held.get("kind") != claimed.get("kind"):
        return False
    if held["kind"] == "name":
        return names_match(str(held["value"]), str(claimed["value"]))
    if held["kind"] == "date" and claimed.get("op", "eq") == "eq":
        # a date as precise as the claim: 2026-03 holds when the fact held 2026-03-24
        return str(held["value"]).startswith(str(claimed["value"])) or str(claimed["value"]).startswith(str(held["value"]))
    op = claimed.get("op", "eq")
    if op == "eq":
        exact = dict(claimed)
        exact.pop("op", None)
        return same_value({k: v for k, v in held.items() if k != "op"}, exact)
    if held["kind"] == "money" and (held.get("currency"), held.get("per")) != (claimed.get("currency"), claimed.get("per")):
        return False
    a, b = _magnitude(held), _magnitude(claimed)
    if a is None or b is None:
        return False
    return {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}.get(op, False)


def value_key(value: dict) -> str:
    """A key one value shares with its other spellings, for counting distinct values."""
    kind = value["kind"]
    if kind == "name":
        return "name:" + " ".join(sorted(re.findall(r"\w+", str(value["value"]).lower()))[-1:])
    m = _magnitude(value)
    extra = f"{value.get('currency', '')}/{value.get('per', '')}" if kind == "money" else ""
    return f"{kind}:{m if m is not None else value.get('value')}{extra}"


# --- observations -----------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    day: int
    value: dict
    weight: float           # how much the observation counts, 0 to 1
    refutes: bool           # a labelled-false row: its value did not hold on its day
    labelled: bool
    row: str
    page: str
    unsure: bool = False    # a labelled-unsure row: the dataset gives no means to tell its values


@dataclass
class ClaimEvidence:
    """What the base says of one claimed value of a fact on a date, without the claim's own row."""

    observations: int = 0
    distinct_values: int = 0
    # the nearest support on each side, outside the claim's page
    before_agrees: float | None = None
    before_gap: float | None = None
    before_labelled: bool = False
    after_agrees: float | None = None
    after_gap: float | None = None
    after_labelled: bool = False
    # weighted support around the date
    agree_before: float = 0.0
    other_before: float = 0.0
    agree_after: float = 0.0
    other_after: float = 0.0
    agree_same_day: float = 0.0
    other_same_day: float = 0.0
    agree_on_page: float = 0.0
    other_on_page: float = 0.0
    refuted_near: float = 0.0
    refuted_other_near: float = 0.0
    agree_ever: bool = False
    # the value held between the nearest supports on each side, if they agree
    bracketed: bool = False
    # rows labelled unsure that state this value, or another
    unsure_agree: float = 0.0
    unsure_other: float = 0.0
    # what the labelled-true rows of other pages fix: the value on the claim's own day, and the value
    # held on both sides of it ("agrees", "contradicts" or None)
    labelled_same_day: str | None = None
    labelled_bracket: str | None = None
    unsure_same_day: bool = False
    labelled_agree: bool = False    # a labelled-true row, of any page and day, states this value

    @property
    def settled_false(self) -> bool:
        """The labelled rows leave no room for the value: on its day, or between two that agree.
        In the practice data this reading is false 98 times out of 98."""
        return self.labelled_same_day == "contradicts" or self.labelled_bracket == "contradicts"

    def features(self, prefix: str = "") -> dict[str, float]:
        f: dict[str, float] = {}
        p = prefix
        if self.unsure_agree:
            f[p + "unsure_agree"] = 1.0
            # unlabelled rows cannot make checkable what the dataset says cannot be checked
            if not self.labelled_agree:
                f[p + "only_unsure_support"] = 1.0
        if self.unsure_other:
            f[p + "unsure_other"] = 1.0
        if self.unsure_same_day:
            f[p + "unsure_same_day"] = 1.0
        if not self.observations:
            f[p + "no_evidence"] = 1.0
            return f
        for name in ("labelled_same_day", "labelled_bracket"):
            reading = getattr(self, name)
            if reading:
                f[f"{p}{name}_{reading}"] = 1.0
        f[p + "distinct_values"] = math.log1p(self.distinct_values)
        for side in ("before", "after"):
            agrees = getattr(self, f"{side}_agrees")
            if agrees is None:
                f[f"{p}{side}_none"] = 1.0
                continue
            gap = getattr(self, f"{side}_gap")
            f[f"{p}{side}_agrees"] = agrees
            f[f"{p}{side}_disagrees"] = 1.0 - agrees
            f[f"{p}{side}_gap"] = math.log1p(gap) / 6
            if getattr(self, f"{side}_labelled"):
                f[f"{p}{side}_labelled_agrees" if agrees >= 0.5 else f"{p}{side}_labelled_disagrees"] = 1.0
        for k in ("agree_before", "other_before", "agree_after", "other_after", "agree_same_day",
                  "other_same_day", "agree_on_page", "other_on_page", "refuted_near", "refuted_other_near"):
            v = getattr(self, k)
            if v:
                f[p + k] = math.log1p(v)
        if self.agree_ever:
            f[p + "agree_ever"] = 1.0
            if not self.agree_before and not self.agree_same_day and self.agree_after:
                f[p + "agree_only_later"] = 1.0
            if not self.agree_after and not self.agree_same_day and self.agree_before and self.other_after:
                f[p + "agree_only_earlier"] = 1.0
        else:
            f[p + "agree_never"] = 1.0
        if self.bracketed:
            f[p + "bracketed"] = 1.0
        # the readings a linear model cannot form from the parts
        before_ok = self.before_agrees is not None and self.before_agrees >= 0.5
        before_bad = self.before_agrees is not None and self.before_agrees < 0.5
        after_ok = self.after_agrees is not None and self.after_agrees >= 0.5
        after_bad = self.after_agrees is not None and self.after_agrees < 0.5
        if before_ok and (after_ok or self.after_agrees is None):
            f[p + "confirmed"] = 1.0
        if before_ok and after_bad:
            f[p + "held_then_changed"] = 1.0          # true until the change, or already stale
        if before_bad and after_ok:
            f[p + "change_announced"] = 1.0           # the first of the new value, or too early
            if self.after_gap is not None and self.after_gap <= 7:
                f[p + "change_confirmed_soon"] = 1.0
        if before_bad and self.agree_before:
            f[p + "stale"] = 1.0                      # a value that held before and no longer does
        if before_bad and not self.agree_before and not self.agree_after and not self.agree_same_day:
            f[p + "contradicted_alone"] = 1.0         # a value nobody else ever states
        if before_bad and self.after_agrees is None and not self.agree_before:
            f[p + "new_value_last"] = 1.0             # a new value nobody confirms yet
        if self.before_agrees is None and after_ok:
            f[p + "first_then_confirmed"] = 1.0
        if self.before_agrees is None and after_bad:
            f[p + "first_then_contradicted"] = 1.0
        if self.before_labelled and before_bad:
            f[p + "labelled_contradicts"] = 1.0
        if self.agree_same_day and not self.other_same_day:
            f[p + "same_day_confirmed"] = 1.0
        if self.other_same_day and not self.agree_same_day:
            f[p + "same_day_contradicted"] = 1.0
        return f


class KnowledgeBase:
    """Every observation of every fact, by day."""

    def __init__(self, near_days: int = 30):
        self.facts: dict[Fact, list[Observation]] = defaultdict(list)
        self._days: dict[Fact, list[int]] = {}
        self.near_days = near_days

    def add(self, fact: Fact, observation: Observation) -> None:
        self.facts[fact].append(observation)
        self._days.pop(fact, None)

    def add_row(self, row: str, page: str, iso_date: str, facts: list[tuple[Fact, dict, float]],
                label: str | None, unlabelled_trust: float = 0.75) -> None:
        """A row's facts, each as (fact, value, resolver confidence), with the row's label if it has one."""
        day = day_number(iso_date)
        for fact, value, confidence in facts:
            if label == "true":
                obs = Observation(day, value, 1.0, False, True, row, page)
            elif label == "false":
                # one false value makes the row false: each of its values is a weak refutation
                obs = Observation(day, value, 1.0 / max(1, len(facts)), True, True, row, page)
            elif label == "unsure":
                # neither a support nor a refutation: a mark that this value cannot be checked
                obs = Observation(day, value, 1.0, False, True, row, page, unsure=True)
            else:
                obs = Observation(day, value, unlabelled_trust * confidence, False, False, row, page)
            self.add(fact, obs)

    def _sorted(self, fact: Fact) -> tuple[list[Observation], list[int]]:
        if fact not in self._days:
            self.facts[fact].sort(key=lambda o: o.day)
            self._days[fact] = [o.day for o in self.facts[fact]]
        return self.facts[fact], self._days[fact]

    def values_of(self, fact: Fact) -> list[Observation]:
        return self._sorted(fact)[0]

    def evidence(self, fact: Fact, iso_date: str, value: dict, row: str, page: str) -> ClaimEvidence:
        obs, days = self._sorted(fact)
        day = day_number(iso_date)
        ev = ClaimEvidence()
        others = [o for o in obs if o.row != row]
        for o in [o for o in others if o.unsure]:
            if values_agree(o.value, value):
                ev.unsure_agree += o.weight
                ev.unsure_same_day |= o.day == day
            else:
                ev.unsure_other += o.weight
        others = [o for o in others if not o.unsure]
        ev.observations = len(others)
        if not others:
            return ev
        ev.distinct_values = len({value_key(o.value) for o in others if not o.refutes})
        exact = {k: v for k, v in value.items() if k != "op"}
        nearest_before: Observation | None = None
        nearest_after: Observation | None = None
        for o in others:
            agrees = values_agree(o.value, value) if not o.refutes else same_value(o.value, exact)
            near = abs(o.day - day) <= self.near_days
            if o.refutes:
                if near:
                    if agrees:
                        ev.refuted_near += o.weight
                    else:
                        ev.refuted_other_near += o.weight
                continue
            if agrees:
                ev.agree_ever = True
                ev.labelled_agree |= o.labelled
            if o.page == page:
                if agrees:
                    ev.agree_on_page += o.weight
                else:
                    ev.other_on_page += o.weight
                continue
            if o.day == day:
                if agrees:
                    ev.agree_same_day += o.weight
                else:
                    ev.other_same_day += o.weight
            elif o.day < day:
                if agrees:
                    ev.agree_before += o.weight
                else:
                    ev.other_before += o.weight
            else:
                if agrees:
                    ev.agree_after += o.weight
                else:
                    ev.other_after += o.weight
            # the nearest trusted support on each side, a labelled row first among equals
            if o.weight >= 0.4:
                if o.day <= day and (nearest_before is None or (o.day, o.labelled) >= (nearest_before.day, nearest_before.labelled)):
                    nearest_before = o
                if o.day >= day and (nearest_after is None or (-o.day, o.labelled) >= (-nearest_after.day, nearest_after.labelled)):
                    nearest_after = o
        if nearest_before is not None:
            ev.before_agrees = self._agreement(others, nearest_before.day, value, page)
            ev.before_gap = day - nearest_before.day
            ev.before_labelled = nearest_before.labelled
        if nearest_after is not None:
            ev.after_agrees = self._agreement(others, nearest_after.day, value, page)
            ev.after_gap = nearest_after.day - day
            ev.after_labelled = nearest_after.labelled
        if nearest_before is not None and nearest_after is not None:
            ev.bracketed = same_value(_plain(nearest_before.value), _plain(nearest_after.value))
        self._settle(ev, others, day, value, page)
        return ev

    @staticmethod
    def _settle(ev: ClaimEvidence, others: list[Observation], day: int, value: dict, page: str) -> None:
        """What the labelled-true rows of other pages fix about the value on this day."""
        sure = [o for o in others if o.labelled and not o.refutes and o.page != page and "op" not in o.value]
        same_day = [o for o in sure if o.day == day]
        if same_day:
            ev.labelled_same_day = "agrees" if any(values_agree(o.value, value) for o in same_day) else "contradicts"
        before = [o for o in sure if o.day < day]
        after = [o for o in sure if o.day > day]
        if before and after:
            last = max(before, key=lambda o: o.day)
            first = min(after, key=lambda o: o.day)
            if same_value(last.value, first.value):
                ev.labelled_bracket = "agrees" if values_agree(last.value, value) else "contradicts"

    @staticmethod
    def _agreement(others: list[Observation], day: int, value: dict, page: str) -> float:
        """The weighted share of the supports on one day that agree with the value."""
        agree = total = 0.0
        for o in others:
            if o.day == day and not o.refutes and o.page != page and o.weight >= 0.4:
                total += o.weight
                if values_agree(o.value, value):
                    agree += o.weight
        return agree / total if total else 0.5


def _plain(value: dict) -> dict:
    return {k: v for k, v in value.items() if k != "op"}
