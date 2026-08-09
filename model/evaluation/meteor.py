"""METEOR for Indonesian: NLTK's METEOR with a Sastrawi stemmer and Wordnet
Bahasa (Open Multilingual WordNet, lang="ind") for the synonym stage.

Why this exists
---------------
NLTK's ``meteor_score`` exposes ``stemmer`` and ``wordnet`` parameters, but its
internal synonym matcher (``_enum_wordnetsyn_match``) calls
``wordnet.synsets(word)`` / ``synset.lemmas()`` / ``lemma.name()`` with *no*
language argument -- so passing the real reader yields the English synsets. We
inject two small pieces:

  * ``SastrawiStemmer`` -- wraps Sastrawi in NLTK's ``StemmerI`` interface so the
    stem-match stage works for Indonesian morphology.
  * ``IndoWordnet`` -- a duck-typed stand-in for ``WordNetCorpusReader`` whose
    ``.synsets()`` / ``.lemmas()`` force ``lang="ind"`` (Wordnet Bahasa).

Everything else -- the least-crossing alignment, chunk counting, fragmentation
penalty, and the alpha/beta/gamma F-mean -- is NLTK's own METEOR machinery,
reused unchanged.

One-time data setup (Wordnet Bahasa ships inside OMW)::

    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Caveats to state in the thesis:
  * Wordnet Bahasa's Indonesian coverage is partial; words with no synset fall
    through to exact + stem matching (still valid, just narrower).
  * NLTK filters multiword lemmas (``kereta_api``) out of the synonym set.
  * alpha/beta/gamma are left at NLTK defaults (tuned on English human
    judgments), kept for comparability rather than re-tuned for Indonesian.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Split into word tokens and standalone punctuation. METEOR expects
# pre-tokenized input; casing is handled by meteor_score's preprocess (str.lower).
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class _Lemma:
    """Minimal stand-in for nltk Lemma: only .name() is used by METEOR."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _IndSynset:
    """Wraps a real WordNet synset so .lemmas() returns Indonesian lemmas."""

    __slots__ = ("_synset",)

    def __init__(self, synset) -> None:
        self._synset = synset

    def lemmas(self) -> list[_Lemma]:
        return [_Lemma(n) for n in self._synset.lemma_names("ind")]


class IndoWordnet:
    """Duck-typed replacement for NLTK's WordNetCorpusReader that resolves
    synonyms from Wordnet Bahasa. METEOR only ever calls
    ``.synsets(word) -> [obj with .lemmas() -> [obj with .name()]]``."""

    def __init__(self, wn) -> None:
        self._wn = wn

    def synsets(self, word: str) -> list[_IndSynset]:
        return [_IndSynset(s) for s in self._wn.synsets(word, lang="ind")]


@lru_cache(maxsize=1)
def _load_backends():
    """Build (SastrawiStemmer, IndoWordnet) once, with actionable errors if a
    dependency or the OMW Indonesian data is missing."""
    try:
        from nltk.corpus import wordnet as wn
        from nltk.stem.api import StemmerI
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "METEOR needs nltk (see requirements.txt), then one-time data:\n"
            '  python -c "import nltk; nltk.download(\'wordnet\'); '
            "nltk.download('omw-1.4')\""
        ) from exc

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "METEOR needs Sastrawi for Indonesian stemming (pip install Sastrawi)."
        ) from exc

    # Force the Indonesian OMW data to resolve now, with a clear message if absent.
    try:
        wn.synsets("rumah", lang="ind")
    except LookupError as exc:
        raise LookupError(
            "Wordnet Bahasa (Indonesian) data not found. Run:\n"
            '  python -c "import nltk; nltk.download(\'wordnet\'); '
            "nltk.download('omw-1.4')\""
        ) from exc

    _sastrawi = StemmerFactory().create_stemmer()

    class SastrawiStemmer(StemmerI):
        def stem(self, token: str) -> str:
            return _sastrawi.stem(token)

    return SastrawiStemmer(), IndoWordnet(wn)


def compute_meteor(hypotheses: list[str], references: list[str]) -> dict:
    """Corpus METEOR (mean of sentence scores) for Indonesian, scaled to 0-100
    to match the BLEU / chrF++ reporting scale."""
    from nltk.translate.meteor_score import single_meteor_score

    stemmer, indo_wn = _load_backends()
    scores = [
        single_meteor_score(
            _tokenize(ref),
            _tokenize(hyp),
            stemmer=stemmer,
            wordnet=indo_wn,
        )
        for hyp, ref in zip(hypotheses, references)
    ]
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"meteor": round(mean * 100, 2)}
