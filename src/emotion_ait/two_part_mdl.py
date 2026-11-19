"""Dataset-level two-part MDL for the 4-class n-gram classifiers.

This module only *measures* fitted classifiers; it never changes how they
predict. The classification rule stays ``argmin_e L(x | M_e)``.

Objective (for a whole training set D with known labels)::

    L_total(D, M) = L(M) + L(D | M)
    L(M)          = sum_e L(M_e)                      paid once, not per comment
    L(D | M)      = sum_i L(x_i | M_{y_i}) = sum_i -log2 P(x_i | M_{y_i})

Model code (deliberately simple, *not* an optimal encoding)
----------------------------------------------------------
``L(M_e) = B * (number of stored count entries of M_e) + B * (number of scalar
parameters)``, with ``B = 32`` bits by default.

*Main convention ("sufficient")*: count the entries of the class's n-gram
count table, i.e. the n-gram types actually observed in that class's training
comments. The counters only hold observed n-grams, so **zero-count entries are
excluded**. Context totals, Kneser-Ney continuation counts and per-context
type counts are not charged: they are deterministic functions of that table
and a decoder can recompute them. Scalars: ``k`` for add-k, one discount per
Kneser-Ney order.

*Alternative convention ("stored")*: charge every count table the
implementation keeps in memory (n-gram counts + context totals for add-k;
for Kneser-Ney, every order's counts, context totals and context type counts).

*Key cost (sensitivity)*: a count is useless without the identity of its
n-gram. ``key_bits`` adds ``n * ceil(log2(V + 1))`` bits per entry (``V`` =
prediction vocabulary; ``+1`` for ``<s>``).

The shared word vocabulary is identical for every configuration trained on the
same comments, so it is reported separately (``vocab_entries``) and left out of
``L(M_e)``; it cannot change which configuration is selected.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

from .mdl_classifier import MDLClassifier
from .single_label_27.mdl import MDL27Classifier

DEFAULT_BITS_PER_ENTRY = 32


def _is_kneser_ney(model: MDLClassifier) -> bool:
    return isinstance(model, MDL27Classifier) and model.smoothing == "kneser_ney"


def count_entries(model: MDLClassifier) -> Dict[int, Dict[str, int]]:
    """Per class: number of stored entries under each convention, plus scalars."""
    if model._per_class_models is None:
        raise RuntimeError("count_entries called before fit().")
    out: Dict[int, Dict[str, int]] = {}
    for c, tables in model._per_class_models.items():
        ngram = tables["ngram_counts"]
        if any(v <= 0 for v in ngram.values()):
            raise ValueError("Count table holds non-positive entries; the convention assumes observed n-grams only.")
        sufficient = len(ngram)
        if _is_kneser_ney(model):
            kn = model._kn["per_class"][c]
            levels = [kn["top"]] + [kn["lower"][m] for m in sorted(kn["lower"])]
            # A pooled back-off is stored once for all classes (see _shared_kn_entries).
            if model.kn_backoff == "shared":
                levels = [kn["top"]]
            stored = sum(len(lv["counts"]) + len(lv["context_totals"]) + len(lv["context_types"]) for lv in levels)
            scalars = 1 + (len(kn["lower"]) if model.kn_backoff == "class" else 0)
        else:
            stored = sufficient + len(tables["context_counts"])
            scalars = 1  # k
        out[c] = {"sufficient": sufficient, "stored": stored, "scalars": scalars}
    return out


def _shared_kn_entries(model: MDLClassifier) -> Dict[str, int]:
    """Entries of a pooled Kneser-Ney back-off, stored once for all classes."""
    if not (_is_kneser_ney(model) and model.kn_backoff == "shared"):
        return {"stored": 0, "scalars": 0}
    shared = model._kn["shared_lower"]
    return {
        "stored": sum(len(lv["counts"]) + len(lv["context_totals"]) + len(lv["context_types"]) for lv in shared.values()),
        "scalars": len(shared),
    }


def model_description_length(model: MDLClassifier, bits_per_entry: int = DEFAULT_BITS_PER_ENTRY) -> Dict[str, float]:
    """``L(M) = sum_e L(M_e)`` in bits under the conventions above."""
    entries = count_entries(model)
    shared = _shared_kn_entries(model)
    n_sufficient = sum(e["sufficient"] for e in entries.values())
    n_stored = sum(e["stored"] for e in entries.values()) + shared["stored"]
    n_scalars = sum(e["scalars"] for e in entries.values()) + shared["scalars"]
    key_bits = model.n * math.ceil(math.log2(model._prediction_vocab_size + 1))
    return {
        "entries_sufficient": n_sufficient,
        "entries_stored": n_stored,
        "scalars": n_scalars,
        "vocab_entries": len(model._word_vocab),
        "L_model_bits": bits_per_entry * (n_sufficient + n_scalars),
        "L_model_stored_bits": bits_per_entry * (n_stored + n_scalars),
        "L_model_with_keys_bits": (bits_per_entry + key_bits) * n_sufficient + bits_per_entry * n_scalars,
        "key_bits_per_entry": key_bits,
    }


def data_description_length(model: MDLClassifier, texts: Sequence[str], labels: Sequence[int]) -> float:
    """``L(D | M) = sum_i L(x_i | M_{y_i})`` in bits, with the true label of each comment."""
    scores = model.description_lengths(list(texts))
    return float(sum(s[int(y)] for s, y in zip(scores, labels)))


def two_part_summary(
    model: MDLClassifier, texts: Sequence[str], labels: Sequence[int], bits_per_entry: int = DEFAULT_BITS_PER_ENTRY
) -> Dict[str, float]:
    """Model cost, data cost and their total for a fitted model on its data."""
    summary = model_description_length(model, bits_per_entry)
    data_bits = data_description_length(model, texts, labels)
    total = summary["L_model_bits"] + data_bits
    summary.update(
        {
            "n_comments": len(texts),
            "L_data_bits": data_bits,
            "L_total_bits": total,
            "L_total_stored_bits": summary["L_model_stored_bits"] + data_bits,
            "L_total_with_keys_bits": summary["L_model_with_keys_bits"] + data_bits,
            "model_share": summary["L_model_bits"] / total,
            "data_bits_per_comment": data_bits / len(texts),
        }
    )
    return summary
