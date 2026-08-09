"""Cleaning rules for parallel sentence pairs.

Kept deliberately case- and punctuation-preserving: the models are evaluated
with sacreBLEU/chrF++ against raw FLORES-200 references, so training data must
keep its natural casing and punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize, strip control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def length_ratio_ok(source: str, target: str, max_ratio: float) -> bool:
    src_len, tgt_len = word_count(source), word_count(target)
    if src_len == 0 or tgt_len == 0:
        return False
    ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
    return ratio <= max_ratio


def longest_word_len(text: str) -> int:
    return max((len(w) for w in text.split()), default=0)


def long_word_ok(source: str, target: str, max_word_length: int) -> bool:
    """OpusFilter LongWordFilter: reject the pair if any word exceeds the limit.

    Catches noise like URLs, random strings, and unsegmented text.
    """
    return max(longest_word_len(source), longest_word_len(target)) <= max_word_length


def symbol_counts_match(source: str, target: str, symbols: str) -> bool:
    """Reject pairs where a structural symbol appears a different number of
    times in the source than in the target.

    A faithful translation preserves brackets 1:1 -- ``(``, ``)``, ``[``, ``]``,
    ``{``, ``}`` carry structure that survives translation, so a count mismatch
    is a reliable signal of a broken or misaligned pair (truncation, merged
    fragments, editorial notes like ``Artinya:`` glued on, orphaned ``]]>``).
    Quotes and apostrophes are deliberately NOT checked here: English
    contractions (``you'll``, ``God's``) and quote conventions (English
    ``"Greek"`` vs Indonesian ``'Greek'``) make their counts differ for
    perfectly good pairs. Counting is done after normalize_unicode, whose NFKC
    pass already folds full-width brackets onto their ASCII forms.
    """
    return all(source.count(sym) == target.count(sym) for sym in symbols)


class LanguageFilter:
    """FastText language-ID filter. Disabled gracefully when the model is missing."""

    def __init__(self, model_path, threshold: float):
        self.threshold = threshold
        self.model = None
        try:
            import fasttext

            self.model = fasttext.load_model(str(model_path))
        except Exception as exc:  # missing wheel or missing lid.176.bin
            print(f"[cleaning] Language filter disabled: {exc}")

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def detect(self, text: str) -> tuple[str, float]:
        """Return the top predicted language code and its confidence."""
        if self.model is None:
            return "", 1.0
        labels, probs = self.model.predict(text.replace("\n", " "))
        return labels[0].replace("__label__", ""), float(probs[0])

    def matches(self, text: str, expected_lang: str) -> bool:
        if self.model is None:
            return True
        lang, prob = self.detect(text)
        return lang == expected_lang and prob >= self.threshold


@dataclass
class CleaningStats:
    total: int = 0
    kept: int = 0
    dropped: dict = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {"total": self.total, "kept": self.kept, "dropped": self.dropped}


class PairCleaner:
    """Applies all cleaning rules to (source, target) pairs and tracks statistics."""

    def __init__(self, cfg: dict, lang_filter: LanguageFilter):
        preprocessing = cfg["preprocessing"]
        self.min_words = preprocessing["min_words"]
        self.max_words = preprocessing["max_words"]
        self.max_ratio = preprocessing["max_length_ratio"]
        self.max_word_length = preprocessing["max_word_length"]
        # Structural symbols whose count must match between source and target.
        # Empty string disables the check. See symbol_counts_match for rationale.
        self.symbol_check = preprocessing.get("symbol_check_chars", "()[]{}")
        self.source_lang = cfg["data"]["source_lang"]
        self.target_lang = cfg["data"]["target_lang"]
        self.lang_filter = lang_filter
        # Store hashes rather than the full (source, target) strings: at tens of
        # millions of pairs a set of string tuples would need tens of GB, while a
        # set of 64-bit hashes stays a few GB. Collision risk at 70M is ~1e-4.
        self.seen: set[int] = set()
        self.stats = CleaningStats()
        # Keep up to this many example pairs per drop reason for inspection.
        self.drop_sample_cap = preprocessing.get("drop_sample_cap", 25)
        self.drop_samples: dict[str, list] = {}

    def _reject(self, reason, source, target, detail=None):
        """Record a drop, stash a few examples, and return None."""
        self.stats.drop(reason)
        bucket = self.drop_samples.setdefault(reason, [])
        if len(bucket) < self.drop_sample_cap:
            record = {"source": source, "target": target}
            if detail:
                record.update(detail)
            bucket.append(record)
        return None

    def clean(self, source: str, target: str) -> tuple[str, str] | None:
        """Return the cleaned pair, or None if it should be dropped."""
        self.stats.total += 1
        source = normalize_unicode(source)
        target = normalize_unicode(target)

        if not source or not target:
            return self._reject("empty", source, target)
        for text in (source, target):
            if not (self.min_words <= word_count(text) <= self.max_words):
                return self._reject("length", source, target)
        if not length_ratio_ok(source, target, self.max_ratio):
            return self._reject("length_ratio", source, target)
        if not long_word_ok(source, target, self.max_word_length):
            return self._reject("long_word", source, target)
        if self.symbol_check and not symbol_counts_match(source, target, self.symbol_check):
            return self._reject("symbol_mismatch", source, target)
        pair_hash = hash((source, target))
        if pair_hash in self.seen:
            return self._reject("duplicate", source, target)
        if self.lang_filter.enabled:
            src_lang, src_prob = self.lang_filter.detect(source)
            if not (src_lang == self.source_lang and src_prob >= self.lang_filter.threshold):
                return self._reject(
                    "source_language", source, target,
                    {"detected": src_lang, "confidence": round(src_prob, 3)},
                )
            tgt_lang, tgt_prob = self.lang_filter.detect(target)
            if not (tgt_lang == self.target_lang and tgt_prob >= self.lang_filter.threshold):
                return self._reject(
                    "target_language", source, target,
                    {"detected": tgt_lang, "confidence": round(tgt_prob, 3)},
                )

        self.seen.add(pair_hash)
        self.stats.kept += 1
        return source, target
