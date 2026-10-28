"""Reads the values a text states, as contracts/value-forms.md writes them.

Each value is read where it stands, whole, with its comparison words, its
unit and the flags that make the referee read it as nothing (a negation
before it in its clause, an approximation, one end of a range).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- vocabulary -------------------------------------------------------------

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _number_words() -> dict[str, int]:
    words = dict(UNITS)
    for t, tv in TENS.items():
        words[t] = tv
        for u, uv in UNITS.items():
            if 1 <= uv <= 9:
                words[f"{t}-{u}"] = tv + uv
    words["a dozen"] = 12
    words["a hundred"] = 100
    for u, uv in UNITS.items():
        if 1 <= uv <= 9:
            words[f"{u} hundred"] = uv * 100
    return words


NUMBER_WORDS = _number_words()
_WORDS_ALT = "|".join(sorted((re.escape(w) for w in NUMBER_WORDS), key=len, reverse=True))

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}

DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
NUM = rf"(?:{DIGITS}|(?:{_WORDS_ALT})(?![\w-]))"
SCALE = r"(?:\s?(?:thousand|million|billion|bn)\b|(?:k|M)\b)"

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"

CURRENCY_SYMBOLS = {"US$": "USD", "€": "EUR", "$": "USD", "£": "GBP"}
CURRENCY_CODES = "EUR USD GBP CHF RON PLN CZK HUF SEK NOK DKK JPY CNY INR CAD AUD".split()
CURRENCY_WORDS = {"euro": "EUR", "euros": "EUR", "dollar": "USD", "dollars": "USD", "lei": "RON"}

PERIODS = {
    "day": "day", "daily": "day", "week": "week", "weekly": "week", "month": "month", "mo": "month",
    "monthly": "month", "year": "year", "yr": "year", "yearly": "year", "annum": "year", "annually": "year",
}

PHYSICAL_UNITS = {
    "B": "B", "KB": "KB", "kB": "KB", "kb": "KB", "MB": "MB", "mb": "MB", "GB": "GB", "gb": "GB",
    "TB": "TB", "tb": "TB", "byte": "B", "kilobyte": "KB", "megabyte": "MB", "gigabyte": "GB",
    "gig": "GB", "terabyte": "TB",
    "mm": "mm", "cm": "cm", "m": "m", "km": "km", "in": "inch", "inch": "inch", "inche": "inch",
    "millimetre": "mm", "millimeter": "mm", "centimetre": "cm", "centimeter": "cm",
    "metre": "m", "meter": "m", "kilometre": "km", "kilometer": "km",
    "g": "g", "kg": "kg", "gram": "g", "gramme": "g", "kilogram": "kg", "kilo": "kg",
    "mAh": "mAh",
    "Hz": "Hz", "kHz": "kHz", "MHz": "MHz", "GHz": "GHz", "hertz": "Hz", "kilohertz": "kHz",
    "megahertz": "MHz", "gigahertz": "GHz",
    "mW": "mW", "W": "W", "kW": "kW", "milliwatt": "mW", "watt": "W", "kilowatt": "kW",
}
# Symbols whose case carries meaning; the rest match in any case.
CASE_SENSITIVE_UNITS = {"m", "M", "mW", "MW", "B", "g", "W", "in"}

DATA_SCALE = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}

DURATION_UNITS = {
    "ms": "ms", "millisecond": "ms",
    "s": "s", "sec": "s", "second": "s",
    "min": "min", "minute": "min",
    "h": "h", "hr": "h", "hour": "h",
    "d": "day", "day": "day",
    "wk": "week", "week": "week", "fortnight": "week",
    "mo": "month", "mth": "month", "month": "month",
    "yr": "year", "year": "year", "decade": "year",
}
DURATION_SECONDS = {"ms": 1e-3, "s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800}
# "a"/"an" count as one only after these words (value-forms.md, "Duration").
ARTICLE_AS_ONE = r"(?:under|within|in|for|over|than|at least|at most|to|only|just|about|around)"

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}
APPROX_BEFORE = [
    "more or less", "close to", "approximately", "approx.", "approx", "roughly", "nearly", "almost",
    "around", "about", "circa", "ca.", "ca", "some", "estimated", "est.", "est", "~", "≈", "±",
]
APPROX_AFTER = ["or so", "give or take", "-ish", "ish"]

OPS_BEFORE = [
    ("no more than", "le"), ("not more than", "le"), ("no less than", "ge"), ("not less than", "ge"),
    ("no fewer than", "ge"), ("no later than", "le"), ("no earlier than", "ge"),
    ("more than", "gt"), ("greater than", "gt"), ("higher than", "gt"), ("in excess of", "gt"),
    ("upwards of", "gt"), ("less than", "lt"), ("fewer than", "lt"), ("lower than", "lt"),
    ("at least", "ge"), ("at most", "le"), ("up to", "le"), ("prior to", "lt"),
    ("over", "gt"), ("above", "gt"), ("exceeding", "gt"), ("under", "lt"), ("below", "lt"),
    ("maximum", "le"), ("max", "le"), ("minimum", "ge"),
]
OPS_BEFORE_DATE = [("before", "lt"), ("after", "gt"), ("by", "le")]
OPS_BEFORE_DURATION = [("within", "le")]
OPS_AFTER = [
    ("or less", "le"), ("or fewer", "le"), ("or under", "le"), ("or below", "le"), ("or lower", "le"),
    ("or shorter", "le"), ("or more", "ge"), ("or higher", "ge"), ("or above", "ge"),
    ("or longer", "ge"), ("or over", "ge"),
]

CLAUSE_BREAK = re.compile(r"[.,;:!?•·|]|\n\s*\n")
INVISIBLE = re.compile("[​‌‍⁠﻿­]")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# Words after a number that are never its counted noun.
NOT_A_NOUN = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with", "from", "per",
    "is", "are", "was", "were", "be", "has", "have", "had", "will", "would", "can", "it", "its",
    "this", "that", "these", "those", "our", "their", "his", "her", "my", "your", "than", "as",
    "more", "less", "new", "now", "so", "too", "each", "every", "all", "times", "x",
    "within", "across", "after", "before", "since", "during", "until", "over", "under", "into",
}


# --- results ----------------------------------------------------------------

@dataclass
class ValueMention:
    """One value the text writes, and how the referee would read it."""

    kind: str
    value: dict                     # as claimed_value.schema.json writes it, op included
    text: str
    start: int
    end: int
    op: str = "eq"
    negated: bool = False           # a negation before it in its clause
    approximate: bool = False
    in_range: bool = False          # one end of a range
    flags: list[str] = field(default_factory=list)

    @property
    def stated(self) -> bool:
        """Whether the referee reads it as a value at all."""
        return not (self.negated or self.approximate or self.in_range or "invisible" in self.flags)

    @property
    def exact(self) -> bool:
        """Whether it can be a claim's value: stated, with no comparison."""
        return self.stated and self.op == "eq"


# --- helpers ----------------------------------------------------------------

def parse_number(s: str) -> float | None:
    s = s.strip().lower()
    if s in NUMBER_WORDS:
        return float(NUMBER_WORDS[s])
    if re.fullmatch(r"\d+\.\d{3}", s):
        return None  # fifteen hundred in much of Europe, or one and a half
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def apply_scale(n: float, scale: str | None) -> float:
    if not scale:
        return n
    key = scale.strip()
    if key == "M":
        return n * 1e6
    return n * SCALES.get(key.lower(), 1)


def clean_number(n: float) -> int | float:
    nearest = round(n)
    if abs(n - nearest) <= 1e-9 * max(1.0, abs(n)):
        return int(nearest)  # 1.2 * 1e6 is 1,200,000, not 1,200,000.0000000002
    return n


def singular(word: str) -> str:
    w = word.lower()
    irregular = {"people": "person", "children": "child", "men": "man", "women": "woman", "data": "data"}
    if w in irregular:
        return irregular[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ches|shes|sses|xes|zes)$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 2:
        return w[:-1]
    return w


def _clause_start(text: str, pos: int) -> int:
    start = 0
    for m in CLAUSE_BREAK.finditer(text, 0, pos):
        # a decimal point or thousands comma inside a number is not a break
        if m.group(0) in ".," and 0 < m.start() < len(text) - 1 and text[m.start() - 1].isdigit() and text[m.start() + 1].isdigit():
            continue
        start = m.end()
    return start


def _words_before(text: str, pos: int) -> str:
    return text[_clause_start(text, pos):pos].lower()


def _negated(text: str, pos: int) -> bool:
    before = _words_before(text, pos)
    for w in WORD.findall(before):
        w = w.lower().replace("’", "'")
        if w in NEGATIONS or w.endswith("n't"):
            # "no more than" and friends are comparison words, not negations
            if w in ("no", "not") and re.search(rf"\b{w} (more|less|fewer|later|earlier) than\s*$", before):
                continue
            return True
    return False


def _ends_with(before: str, phrase: str) -> bool:
    if phrase.isalpha() or " " in phrase:
        return re.search(rf"(?<![\w]){re.escape(phrase)}\s*$", before) is not None
    return before.rstrip().endswith(phrase)


def _op_before(text: str, pos: int, kind: str) -> str:
    before = text[max(0, pos - 40):pos].lower()
    table = list(OPS_BEFORE)
    if kind == "date":
        table = OPS_BEFORE_DATE + [(p, o) for p, o in table if p in ("no later than", "no earlier than", "prior to")]
    elif kind == "duration":
        table = OPS_BEFORE_DURATION + table
    for phrase, op in sorted(table, key=lambda p: len(p[0]), reverse=True):
        if _ends_with(before, phrase):
            return op
    return "eq"


def _op_after(text: str, end: int) -> str:
    after = text[end:end + 20].lower()
    if after.startswith("+"):
        return "ge"
    for phrase, op in OPS_AFTER:
        if re.match(rf"\s*{re.escape(phrase)}\b", after):
            return op
    return "eq"


def _approximate(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 25):start].lower()
    after = text[end:end + 15].lower()
    if any(_ends_with(before, p) for p in APPROX_BEFORE):
        return True
    return any(re.match(rf"\s*{re.escape(p)}\b" if p[0].isalpha() else re.escape(p), after) for p in APPROX_AFTER)


def _in_range(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 30):start]
    after = text[end:end + 30]
    if re.search(r"\bbetween\s+$", before, re.I) or re.match(r"\s*and\s+\S", after) and re.search(r"\bbetween\s+[^,;]*$", before, re.I):
        return True
    if re.match(r"\s*(?:[-–—]|to)\s*[€$£]?\d", after) or re.search(r"\d\s*(?:[-–—]|to)\s*$", before):
        return True
    if re.match(r"\s*or\s+\d+\s*\w+", after) and not re.match(r"\s*or\s+[€$£]", after):
        # "1 or 2 days" is a range; "€49 a month or €490 a year" is a choice
        return True
    if re.search(r"\d\s+or\s+$", before):
        return True
    return False


# --- the reader -------------------------------------------------------------

class ValueReader:
    """Finds the values a text states, without reading inside masked spans (names)."""

    def __init__(self, counted_nouns: set[str] | None = None):
        # nouns the dataset counts, used to pick the noun among the words after a number
        self.counted_nouns = counted_nouns or set()

    def read(self, text: str, masked: list[tuple[int, int]] | None = None) -> list[ValueMention]:
        self._text = text
        self._taken: list[tuple[int, int]] = list(masked or [])
        self._masked = list(masked or [])
        found: list[ValueMention] = []
        for reader in (self._dates, self._money, self._percents, self._versions, self._durations, self._quantities):
            found.extend(reader())
        for v in found:
            self._qualify(v)
        return sorted(found, key=lambda v: v.start)

    # each reader claims spans so later readers do not read them again
    def _free(self, start: int, end: int) -> bool:
        return not any(start < e and s < end for s, e in self._taken)

    def _claim(self, start: int, end: int) -> None:
        self._taken.append((start, end))

    def _mention(self, kind: str, value: dict, m_start: int, m_end: int, op: str | None = None) -> ValueMention | None:
        if not self._free(m_start, m_end):
            return None
        self._claim(m_start, m_end)
        return ValueMention(kind=kind, value=value, text=self._text[m_start:m_end], start=m_start, end=m_end, op=op or "eq")

    def _qualify(self, v: ValueMention) -> None:
        t = self._text
        if v.op == "eq":
            v.op = _op_before(t, v.start, v.kind)
        if v.op == "eq":
            v.op = _op_after(t, v.end)
        if v.op != "eq":
            v.value["op"] = v.op
        v.negated = _negated(t, v.start)
        v.approximate = _approximate(t, v.start, v.end)
        v.in_range = v.kind not in ("date", "version", "name") and _in_range(t, v.start, v.end)
        if INVISIBLE.search(t[max(0, v.start - 1):v.end + 1]):
            v.flags.append("invisible")

    # dates ------------------------------------------------------------------
    def _dates(self) -> list[ValueMention]:
        out = []
        t = self._text
        patterns = [
            (rf"\b(\d{{4}})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", "ymd"),
            (rf"\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s*({MONTH})\s+(\d{{4}})\b", "dmy"),
            (rf"\b({MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", "mdy"),
            (rf"\b(\d{{4}})-(0[1-9]|1[0-2])\b(?!-\d)", "ym"),
            (rf"\b({MONTH})\s+(\d{{4}})\b", "my"),
            (r"(?<![\w$€£.,/-])((?:19|20)\d{2})(?![\w.,%/-]|\s?(?:%|percent|per cent))", "y"),
        ]
        for pattern, shape in patterns:
            for m in re.finditer(pattern, t):
                if shape == "y" and re.search(r"\b[QH][1-4]\s*$", t[:m.start()]):
                    continue  # quarters and halves are not dates
                value = self._date_value(shape, m)
                if value is None:
                    continue
                v = self._mention("date", {"kind": "date", "value": value}, m.start(), m.end())
                if v:
                    out.append(v)
        return out

    @staticmethod
    def _date_value(shape: str, m: re.Match) -> str | None:
        month = lambda s: MONTHS.get(s.rstrip(".").lower())
        if shape == "ymd":
            return f"{m[1]}-{m[2]}-{m[3]}"
        if shape == "ym":
            return f"{m[1]}-{m[2]}"
        if shape == "y":
            return m[1]
        if shape == "dmy":
            d, mo, y = int(m[1]), month(m[2]), m[3]
        elif shape == "mdy":
            mo, d, y = month(m[1]), int(m[2]), m[3]
        else:
            mo = month(m[1])
            return f"{m[2]}-{mo:02d}" if mo else None
        if not mo or not 1 <= d <= 31:
            return None
        return f"{y}-{mo:02d}-{d:02d}"

    # money ------------------------------------------------------------------
    def _money(self) -> list[ValueMention]:
        out = []
        t = self._text
        price = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+,\d{2}(?!\d)|\d+(?:\.\d+)?"
        codes = "|".join(CURRENCY_CODES)
        words = "|".join(CURRENCY_WORDS)
        # the referee reads only "per user" or "/seat" before the period; the input's
        # own sentences also write "per terabyte each month", which is read leniently
        period = (
            r"(?:\s*(?:per\s+user|per\s+seat|/\s*user|/\s*seat|per\s+(?!day|week|month|year|annum)[a-z]+))?"
            r"(?:\s*(?:/\s*|per\s+|an?\s+|each\s+)(?P<per>day|week|month|mo|year|yr|annum)\b"
            r"|\s+(?P<per2>daily|weekly|monthly|yearly|annually))?"
        )
        before = rf"(?P<cur>US\$|[€$£]|\b(?:{codes})\s?)(?P<num>{price})(?P<scale>{SCALE})?"
        after = rf"\b(?P<num>{price})(?P<scale>{SCALE})?\s?(?P<cur>(?:{codes})\b|(?:{words})\b|€)"
        for pattern in (before + period, after + period):
            for m in re.finditer(pattern, t, re.IGNORECASE):
                cur_raw = m["cur"].strip()
                currency = CURRENCY_SYMBOLS.get(cur_raw) or CURRENCY_WORDS.get(cur_raw.lower()) or cur_raw.upper()
                if currency not in CURRENCY_CODES:
                    continue
                num = m["num"]
                if re.fullmatch(r"\d+,\d{2}", num):
                    num = num.replace(",", ".")
                n = parse_number(num)
                if n is None:
                    continue
                scale = m["scale"]
                if scale and scale.strip() == "m":
                    scale = None  # a lower-case m is a metre
                value = {"kind": "money", "value": clean_number(apply_scale(n, scale)), "currency": currency}
                per = m["per"] or m["per2"]
                inferred = None
                if per:
                    value["per"] = PERIODS[per.lower()]
                else:
                    inferred = self._period_before(t, m.start())
                    if inferred:
                        value["per"] = inferred
                v = self._mention("money", value, m.start(), m.end())
                if v:
                    if inferred:
                        v.flags.append("period_inferred")
                    out.append(v)
        return out

    @staticmethod
    def _period_before(text: str, start: int) -> str | None:
        """The period a price's clause names before it: "its monthly price rose to €59".

        The referee reads a period only after the price, so a value read this way
        is flagged `period_inferred`: true to the sentence, but not stated in a form
        the referee reads, and never to be relied on in a claim's text.
        """
        before = text[_clause_start(text, start):start]
        m = re.search(r"\b(monthly|weekly|daily|yearly|annual|annually)\s+(?:[a-z]+\s+){0,3}$", before, re.IGNORECASE)
        if not m:
            return None
        return {"annual": "year", "annually": "year"}.get(m.group(1).lower()) or PERIODS[m.group(1).lower()]

    # percents ---------------------------------------------------------------
    def _percents(self) -> list[ValueMention]:
        out = []
        for m in re.finditer(rf"(?<![\w.,])(?P<num>{NUM})\s?(?:%|percent\b|per\s+cent\b)", self._text, re.IGNORECASE):
            n = parse_number(m["num"]) if not re.fullmatch(r"\d+\.\d{3}", m["num"]) else float(m["num"])
            if n is None:
                continue
            v = self._mention("percent", {"kind": "percent", "value": clean_number(n)}, m.start(), m.end())
            if v:
                out.append(v)
        return out

    # versions ---------------------------------------------------------------
    def _versions(self) -> list[ValueMention]:
        out = []
        t = self._text
        for m in re.finditer(r"\b(?:v|version\s?|ver\.\s?)(\d+(?:\.\d+){0,3})\b(?!\.\d)", t, re.IGNORECASE):
            v = self._mention("version", {"kind": "version", "value": m[1]}, m.start(), m.end())
            if v:
                out.append(v)
        for m in re.finditer(r"(?<![\w.,])(\d+\.\d+\.\d+(?:\.\d+)?)(?![\w.]|\.\d)", t):
            v = self._mention("version", {"kind": "version", "value": m[1]}, m.start(), m.end())
            if v:
                out.append(v)
        # a number with one point straight after a capitalised name: "Hooly Gemini 2.2"
        for m in re.finditer(r"\b[A-Z][\w/]{0,40} (\d+\.\d+)(?!\w|\.\d|\s?(?:%|percent))", t):
            if self._unit_follows(t, m.end(1)):
                continue  # "4.1 stars", "1.6 GB": a quantity, not a version
            v = self._mention("version", {"kind": "version", "value": m[1]}, m.start(1), m.end(1))
            if v:
                out.append(v)
        return out

    # durations --------------------------------------------------------------
    def _durations(self) -> list[ValueMention]:
        out = []
        t = self._text
        unit_alt = r"milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|fortnights?|months?|mos?|mths?|years?|yrs?|decades?"
        pattern = rf"(?<![\w.,$€£])(?P<num>{NUM})(?:\s?-\s?|\s?)(?P<unit>{unit_alt})\b(?![-]?\s?(?:old\b))"
        for m in re.finditer(pattern, t, re.IGNORECASE):
            unit_raw = m["unit"]
            if len(unit_raw) == 1 and unit_raw != unit_raw.lower():
                continue  # a single-letter unit is written in lower case
            if unit_raw.lower() in ("s", "h", "d", "ms") and not re.fullmatch(DIGITS, m["num"]):
                continue
            if unit_raw.lower() in ("mo", "mos") and m["num"].lower() in NUMBER_WORDS:
                continue
            n = parse_number(m["num"])
            if n is None:
                continue
            v = self._duration(n, unit_raw, m.start(), m.end())
            if v:
                out.append(v)
        # "a" or "an" counts as one only after a few words: "under a second", not "a day"
        pattern = rf"\b{ARTICLE_AS_ONE}\s+(an?\s+(?P<unit>second|minute|hour|day|week|fortnight|month|year|decade))\b"
        for m in re.finditer(pattern, t, re.IGNORECASE):
            v = self._duration(1, m["unit"], m.start(1), m.end(1))
            if v:
                out.append(v)
        return out

    def _duration(self, n: float, unit_raw: str, start: int, end: int) -> ValueMention | None:
        key = unit_raw.lower()
        key = DURATION_UNITS.get(key) or DURATION_UNITS.get(singular(key)) or DURATION_UNITS.get(key.rstrip("s"))
        if not key:
            return None
        if unit_raw.lower().startswith("fortnight"):
            n *= 2
        if unit_raw.lower().startswith("decade"):
            n *= 10
        return self._mention("duration", {"kind": "duration", "value": clean_number(n), "unit": key}, start, end)

    # quantities -------------------------------------------------------------
    def _quantities(self) -> list[ValueMention]:
        out = []
        t = self._text
        physical = "|".join(sorted((re.escape(u) for u in PHYSICAL_UNITS), key=len, reverse=True))
        pattern = rf"(?<![\w.,$€£])(?P<num>{NUM})(?P<scale>{SCALE})?(?:\s?-\s?|\s?)(?P<unit>(?:{physical})s?\b|[\"″])"
        for m in re.finditer(pattern, t, re.IGNORECASE):
            unit_raw = m["unit"]
            unit = self._physical_unit(unit_raw)
            if not unit:
                continue
            if unit_raw == "in" and re.match(r"\s+[a-z]", t[m.end():]) and not re.match(r"\s+(?:screen|display)\b", t[m.end():]):
                continue  # "1.5 in production": the preposition, not an inch
            n = parse_number(m["num"])
            if n is None:
                continue
            n = apply_scale(n, m["scale"])
            v = self._mention("quantity", {"kind": "quantity", "value": clean_number(n), "unit": unit}, m.start(), m.end())
            if v:
                out.append(v)
        # counted nouns: up to two words between the number and its noun
        pattern = rf"(?<![\w.,$€£/:-])(?P<num>{NUM})(?P<scale>{SCALE})?(?P<plus>\+)?(?P<rest>(?:\s+[A-Za-z][\w'-]*){{1,3}})"
        for m in re.finditer(pattern, t):
            raw = m["num"]
            if re.fullmatch(r"(?:19|20)\d{2}", raw):
                continue  # a plain four-digit number from 1900 to 2099 is a year
            n = parse_number(raw)
            if n is None:
                continue
            noun = self._counted_noun(m["rest"])
            if not noun:
                continue
            word, offset = noun
            end = m.start("rest") + offset + len(word)
            n = apply_scale(n, m["scale"] if not (m["scale"] or "").strip() == "m" else None)
            unit = singular(word)
            if unit in DURATION_UNITS:
                continue
            v = self._mention("quantity", {"kind": "quantity", "value": clean_number(n), "unit": unit}, m.start(), end)
            if v:
                if m["plus"]:
                    v.op = "ge"
                out.append(v)
        return out

    def _unit_follows(self, text: str, end: int) -> bool:
        nxt = re.match(r"\s?-?\s?([A-Za-z]+|[\"″])", text[end:])
        if not nxt:
            return False
        word = nxt.group(1)
        if word == "in" and not re.match(r"\s?in\b\s*(?:[.,;:!?)]|$|screen|display)", text[end:]):
            return False  # "Hooly Gemini 1.5 in production"
        return (self._physical_unit(word) is not None or singular(word) in self.counted_nouns
                or singular(word) in DURATION_UNITS or word.lower() in DURATION_UNITS)

    @staticmethod
    def _physical_unit(raw: str) -> str | None:
        if raw in ('"', "″"):
            return "inch"
        for candidate in (raw, raw[:-1] if raw.endswith("s") and len(raw) > 2 else None):
            if not candidate:
                continue
            if candidate in PHYSICAL_UNITS:
                return PHYSICAL_UNITS[candidate]
            if candidate not in CASE_SENSITIVE_UNITS:
                for key, unit in PHYSICAL_UNITS.items():
                    if key not in CASE_SENSITIVE_UNITS and key.lower() == candidate.lower():
                        return unit
        return None

    def _counted_noun(self, rest: str) -> tuple[str, int] | None:
        words = [(w.group(0), w.start()) for w in re.finditer(r"[A-Za-z][\w'-]*", rest)][:3]
        # "10,000 of our customers": the connective words come before the noun
        while words and words[0][0].lower() in ("of", "our", "their", "its", "the"):
            words = words[1:]
        # the run of words that can stand before the noun ends at a function word
        run = []
        for word, offset in words:
            if word.lower() in NOT_A_NOUN or word[0].isupper():
                break
            run.append((word, offset))
        if not run:
            return None
        # the last noun the dataset counts wins: "4 data centre sites" counts sites
        known = [(w, o) for w, o in run if singular(w) in self.counted_nouns]
        if known:
            # the first plural noun the dataset counts ("1,800 customers teams"),
            # else the last ("4 data centre sites")
            plural = [(w, o) for w, o in known if singular(w) != w.lower()]
            return plural[0] if plural else known[-1]
        nouns = [(w, o) for w, o in run if not w.lower().endswith("ly")]  # "monthly users"
        return nouns[-1] if len(nouns) > 1 and singular(nouns[-1][0]) != nouns[-1][0].lower() else (nouns or run)[0]


def same_value(a: dict, b: dict) -> bool:
    """Whether two claimed values are one value, after units convert."""
    if a.get("kind") != b.get("kind") or a.get("op", "eq") != b.get("op", "eq"):
        return False
    kind = a["kind"]
    if kind == "quantity":
        ua, ub = a["unit"], b["unit"]
        if ua in DATA_SCALE and ub in DATA_SCALE:
            return abs(a["value"] * DATA_SCALE[ua] - b["value"] * DATA_SCALE[ub]) < 1e-6
        return singular(ua) == singular(ub) and a["value"] == b["value"]
    if kind == "duration":
        ua, ub = a["unit"], b["unit"]
        if ua in DURATION_SECONDS and ub in DURATION_SECONDS:
            return abs(a["value"] * DURATION_SECONDS[ua] - b["value"] * DURATION_SECONDS[ub]) < 1e-9
        if {ua, ub} <= {"month", "year"}:
            return a["value"] * (12 if ua == "year" else 1) == b["value"] * (12 if ub == "year" else 1)
        return ua == ub and a["value"] == b["value"]
    if kind == "money":
        return (a["value"], a["currency"], a.get("per")) == (b["value"], b["currency"], b.get("per"))
    if kind == "name":
        return re.sub(r"\s+", "", str(a["value"])).lower() == re.sub(r"\s+", "", str(b["value"])).lower()
    return a["value"] == b["value"]
