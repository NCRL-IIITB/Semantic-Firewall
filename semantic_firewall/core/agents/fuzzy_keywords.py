"""Levenshtein-based repair of obfuscated keywords before regex matching.

Attackers evade keyword regexes with typos and leetspeak ("ign0re previuos
instrucitons"). ``KeywordRepairer`` rewrites each token that is within a small
edit distance of a known keyword (after mapping leetspeak digits) to that
keyword, so the unchanged regex patterns can match the repaired text.
Callers only *add* matches found on the repaired text; matches on the original
text are unaffected.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_TOKEN = re.compile(r"[A-Za-z0-9@$]{5,}")
_REGEX_ESCAPE = re.compile(r"\\[A-Za-z]")
_WORD = re.compile(r"[A-Za-z]{5,}")


def levenshtein(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """Edit distance (insert/delete/substitute, cost 1 each), Wagner-Fischer.

    With ``max_dist`` set, returns ``max_dist + 1`` as soon as the distance is
    known to exceed it.
    """
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(
                prev[j] + 1,                            # deletion
                cur[j - 1] + 1,                         # insertion
                prev[j - 1] + (a[i - 1] != b[j - 1]),   # substitution
            )
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def max_edits(length: int) -> int:
    """Allowed edits: 1 for words of 5-7 letters, 2 for 8 or more."""
    return 1 if length <= 7 else 2


def keywords_from_patterns(patterns: Iterable[str]) -> frozenset[str]:
    """Literal words of 5+ letters that appear in regex pattern strings."""
    words: set[str] = set()
    for pattern in patterns:
        words.update(w.lower() for w in _WORD.findall(_REGEX_ESCAPE.sub(" ", pattern)))
    return frozenset(words)


class KeywordRepairer:
    def __init__(self, keywords: Iterable[str]):
        self.keywords = frozenset(k.lower() for k in keywords if len(k) >= 5)
        self._by_length: dict[int, list[str]] = {}
        for keyword in sorted(self.keywords):
            self._by_length.setdefault(len(keyword), []).append(keyword)
        self._fix_token = lru_cache(maxsize=8192)(self._fix_token_uncached)

    def _fix_token_uncached(self, token: str) -> Optional[str]:
        lowered = token.lower()
        if lowered in self.keywords:
            return None
        candidate = lowered.translate(_LEET)
        if candidate in self.keywords:
            return candidate
        limit = max_edits(len(candidate))
        best, best_dist = None, limit + 1
        for length in range(len(candidate) - limit, len(candidate) + limit + 1):
            for keyword in self._by_length.get(length, ()):
                dist = levenshtein(candidate, keyword, limit)
                if dist < best_dist:
                    best, best_dist = keyword, dist
        return best

    def repair(self, text: str) -> str:
        """Return ``text`` with near-miss keywords replaced by the keyword."""
        def substitute(match: re.Match[str]) -> str:
            fixed = self._fix_token(match.group(0))
            return fixed if fixed is not None else match.group(0)

        return _TOKEN.sub(substitute, text)
