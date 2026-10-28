"""The lexicon: subjects, their aliases and attributes, and the events.

Names match as whole words, in any case, with the spaces between their words
optional (contracts/claim-forms.md, "Names"). A name inside a longer name
names the longer one only, so matching is longest-first and non-overlapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


def normalize_name(name: str) -> str:
    """The key two spellings of one name share: lower case, no spaces."""
    return re.sub(r"\s+", "", name).lower()


def name_pattern(surface: str) -> re.Pattern:
    """A whole-word, case-insensitive pattern for a name, spaces optional."""
    tokens = surface.split()
    body = r"\s*".join(re.escape(t) for t in tokens)
    # "Hooly Android 16" does not name itself inside the version "Hooly Android 16.0"
    return re.compile(rf"(?<![\w]){body}(?![\w]|\.\d)", re.IGNORECASE)


@dataclass(frozen=True)
class Subject:
    name: str
    aliases: tuple[str, ...]
    attributes: tuple[str, ...]
    is_event: bool = False

    @property
    def surfaces(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class NameMatch:
    """A span of text that names something."""

    key: str          # canonical name of what is named
    surface: str      # the lexicon spelling that matched
    text: str         # the text as written
    start: int
    end: int
    via: str          # "name" or "alias"


@dataclass
class Lexicon:
    subjects: dict[str, Subject]
    events: tuple[str, ...]
    _by_key: dict[str, str] = field(default_factory=dict, repr=False)
    _patterns: list[tuple[re.Pattern, str, str, str]] = field(default_factory=list, repr=False)

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        events = tuple(raw.get("events", []))
        event_keys = {normalize_name(e) for e in events}
        subjects = {
            s["name"]: Subject(
                name=s["name"],
                aliases=tuple(s.get("aliases", [])),
                attributes=tuple(s.get("attributes", [])),
                is_event=normalize_name(s["name"]) in event_keys,
            )
            for s in raw["subjects"]
        }
        lexicon = cls(subjects=subjects, events=events)
        lexicon._index()
        return lexicon

    def _index(self) -> None:
        entries = []
        for s in self.subjects.values():
            for surface in s.surfaces:
                self._by_key.setdefault(normalize_name(surface), s.name)
                entries.append((surface, s.name, "name" if surface == s.name else "alias"))
        entries.sort(key=lambda e: len(normalize_name(e[0])), reverse=True)
        self._patterns = [(name_pattern(surface), surface, key, via) for surface, key, via in entries]

    def resolve(self, name: str) -> Subject | None:
        """The subject a name or alias names, in any case and spacing."""
        key = self._by_key.get(normalize_name(name))
        return self.subjects.get(key) if key else None

    def has_attribute(self, subject: str, attribute: str) -> bool:
        s = self.resolve(subject)
        return s is not None and attribute in s.attributes

    def find(self, text: str) -> list[NameMatch]:
        """Every subject the text names, longest name first, without overlaps."""
        return find_names(text, self._patterns)


def find_names(text: str, patterns: list[tuple[re.Pattern, str, str, str]]) -> list[NameMatch]:
    """Longest-first, non-overlapping matches of `(pattern, surface, key, via)`."""
    taken: list[tuple[int, int]] = []
    found: list[NameMatch] = []
    for pattern, surface, key, via in patterns:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append(NameMatch(key, surface, m.group(0), m.start(), m.end(), via))
    return sorted(found, key=lambda n: n.start)
