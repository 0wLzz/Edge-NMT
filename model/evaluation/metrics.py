"""Translation quality metrics: sacreBLEU BLEU / chrF++, and (optional) METEOR.

BLEU and chrF++ operate on detokenized text against unmodified references, so
scores are comparable with published FLORES-200 results. METEOR is Indonesian-
aware (Sastrawi stemmer + Wordnet Bahasa) and lives in ``meteor.py``; it is
imported lazily so nltk/Sastrawi are only required when it is switched on.
"""

from __future__ import annotations

import sacrebleu


def compute_bleu(hypotheses: list[str], references: list[str]) -> dict:
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {"bleu": round(result.score, 2), "bleu_signature": str(result.format())}


def compute_chrf(hypotheses: list[str], references: list[str]) -> dict:
    # word_order=2 makes this chrF++.
    result = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {"chrf": round(result.score, 2)}


def compute_all(
    hypotheses: list[str],
    references: list[str],
    *,
    chrf: bool = True,
    meteor: bool = False,
) -> dict:
    """BLEU always; chrF++ and METEOR gated by flags (see config `evaluation`)."""
    metrics = {}
    metrics.update(compute_bleu(hypotheses, references))
    if chrf:
        metrics.update(compute_chrf(hypotheses, references))
    if meteor:
        from model.evaluation.meteor import compute_meteor

        metrics.update(compute_meteor(hypotheses, references))
    return metrics
