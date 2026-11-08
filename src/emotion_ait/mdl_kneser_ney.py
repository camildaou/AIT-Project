"""Kneser-Ney MDL for the 4-class track.

The 4-class ablation (``emotion_ait.ablation``) only varied add-``k``
smoothing. Interpolated Kneser-Ney was introduced later, for the 27-class
track, in :class:`emotion_ait.single_label_27.mdl.MDL27Classifier`. This
module reuses that implementation unchanged and only swaps the label set
for the four 4-class emotions (``anger=0, fear=1, joy=2, sadness=3``), so
the two tracks use exactly the same smoothing code:

* same tokenization, shared training-only vocabulary and
  ``<s>``/``</s>``/``<unk>`` conventions as the baseline ``mdl``;
* score ``L(x | M_e)`` (no prior), argmin, ties to the lowest class id;
* interpolated Kneser-Ney of any order, per-class (``"class"``) or pooled
  (``"shared"``) lower orders, discounts estimated per level unless fixed.

The baseline ``mdl_classifier`` and the 27-class module are not modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .single_label_27.mdl import MDL27Classifier

FOUR_CLASS_NAMES: List[str] = ["anger", "fear", "joy", "sadness"]

#: (model_name, n, kn_backoff) of every 4-class Kneser-Ney configuration.
KN_CONFIGS: List[Tuple[str, int, str]] = [
    ("mdl_n2_knclass", 2, "class"),
    ("mdl_n2_knshared", 2, "shared"),
    ("mdl_n3_knclass", 3, "class"),
    ("mdl_n4_knclass", 4, "class"),
]


class MDLKneserNey4Classifier(MDL27Classifier):
    """4-class conditional MDL with interpolated Kneser-Ney smoothing."""

    def __init__(self, n: int = 2, kn_backoff: str = "class", kn_discount: Optional[float] = None) -> None:
        super().__init__(n=n, smoothing="kneser_ney", use_prior=False, kn_backoff=kn_backoff, kn_discount=kn_discount)
        self.class_names = list(FOUR_CLASS_NAMES)


def kn_config_summary(n: int, kn_backoff: str, kn_discount: Optional[float] = None) -> Dict[str, object]:
    """Per-run configuration saved with the metrics."""
    return {
        "kn_backoff": kn_backoff,
        "kn_discount": "estimated" if kn_discount is None else float(kn_discount),
        "n": int(n),
        "smoothing": "kneser_ney",
    }
