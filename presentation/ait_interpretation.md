# Emotions as codes: an algorithmic-information reading of our results

Audience: students and instructor of the Algorithmic Information Theory course.
Every number below was recomputed from `results/metrics/*.csv`, `results/predictions/*.csv` and
`results/diagnostics/*.json`; experiment IDs (E01–E30) refer to `EXPERIMENT_LOG.md`. "Macro-F1" and
"accuracy" are test-set values averaged over the 5 seeds of a training-size ladder unless stated
otherwise; κ is Cohen's chance-corrected agreement on the test set.

## 1. Framework: classification as compression

**Kolmogorov complexity.** K(x) is the length of the shortest program that prints x on a fixed
universal machine; K(x | y) is the same with y given as auxiliary input. K is uncomputable, but every
*real* compressor gives an upper bound: if a decodable code C describes x in |C(x)| bits, then
K(x) ≤ |C(x)| + O(1).

**From codes to probabilities (Kraft).** A prefix code with lengths ℓ(x) satisfies Σ 2^(−ℓ(x)) ≤ 1;
conversely any probability model P defines a prefix code with ℓ(x) = ⌈−log₂ P(x)⌉. So "a language
model" and "a compressor" are the same object, and code lengths are measured in bits.

**MDL as the computable stand-in.** For each emotion e we build a code M_e (a smoothed n-gram model
estimated only on texts of class e) and measure the conditional description length

  L(x | M_e) = −Σᵢ log₂ P(wᵢ | wᵢ₋ₙ₊₁ … wᵢ₋₁, M_e)  (bits),

then predict ê = argmin_e L(x | M_e): *the class whose code compresses the text best*. L(x | M_e) is a
computable proxy for K(x | e), the information in x that is not already explained by "x expresses
emotion e". This is compression-based classification in the spirit of Li & Vitányi (and of the
normalized compression distance, NCD): instead of a general-purpose compressor applied to
concatenations, we use one explicit statistical code per class, which lets us control and inspect
the code exactly.

**What we approximate and what we don't.** We score only L(x | M_e), not the two-part code
L(M_e) + L(x | M_e) (Section 3), and our models are bigrams-to-4-grams over words — a very small,
fixed model class. Everything below should be read as "MDL relative to this model class".

## 2. Smoothing is coding

A count-based n-gram model must assign a codeword to events it never saw; smoothing *is* the choice of
code for the unseen.

**Add-k (Laplace-type) codes.** P(w | v) = (c(v,w) + k) / (c(v) + kV) spreads a fixed pseudo-count
k over all V ≈ 18.7k vocabulary items (27-class track). With k = 1 that is ~18.7k pseudo-counts per
context, which swamps the real counts of small classes: their codes become nearly uniform, and a
class whose code is "less wrong on average" — the frequent classes — wins. The 27-class add-1 model
(E16) predicts *admiration* 890 times on a test set where it occurs 348 times (seed 42, full size),
and 5 rare classes get F1 = 0.

**Why a single k cannot fix it (E16–E19).** Lowering k removes the bias but makes every unseen
bigram cost ≈ log₂((c + kV)/k) bits — very expensive — and short comments are full of unseen
bigrams. Validation macro-F1 falls monotonically: k = 1 → 0.229, 0.1 → 0.225, 0.01 → 0.215,
0.001 → 0.187 (mean over all sizes). At k = 0.001 the class counts are right (admiration 346 vs 348)
but accuracy drops from 0.428 to 0.304 at full size. One global parameter cannot express "unseen
events are cheap if the word is a plausible continuation in general, expensive otherwise".

**Kneser-Ney as a better universal code (E20).** Interpolated KN subtracts a discount D (estimated from
the count-of-counts, D = n₁ / (n₁ + 2n₂)) from each seen count and gives the freed mass to a *continuation* distribution (in how many distinct contexts
a word occurs), recursing down to the uniform 1/V. This is still a proper code (we verified that every
conditional distribution sums to 1), but it spends its "unseen" budget where unseen events are
likely. Result: best validation macro-F1 of all 27-class MDL variants (0.266 overall), test
macro-F1 0.304 at full size vs 0.279 for add-1, predicted class counts close to the truth (admiration
337 vs 348) and only 2 classes with F1 = 0. The cost: accuracy and κ are lower than add-1 at large
sizes (0.392 vs 0.428; κ 0.354 vs 0.379), because KN reshapes the codes of frequent classes with
distinctive vocabulary.

**Shared back-off is worse (E21).** Pooling the lower-order distribution across all classes
(validation 0.226) makes the 27 codes less distinctive — the information that separates the classes
is partly erased by a shared component.

## 3. Two-part vs conditional MDL, and the code for the label

**Per-text model cost collapses (E05).** Adding L(M_e), the bits needed to transmit class e's n-gram
table, to every text's score makes the classifier predict *fear* — the class with the smallest
table — for all 391 test texts (seed 42, full size); macro-F1 0.071 at every size. In two-part MDL,
L(M) is paid **once for the whole dataset**, so per datum it is amortized to almost nothing; charging
it per text confuses "which hypothesis best explains the corpus" with "which class explains this
text". The negative result is itself instructive: model complexity must be measured on data, not
added to each test point.

**Prequential coding as the honest complexity measure (E07).** Encoding the training set
sequentially — predict each text with the model fitted on the texts before it, then update — gives a
complete code for the data that implicitly charges for model complexity (a complex model pays while
it is still learning). On the 4-class ladder the prequential text code length grows from 12,227 bits
(n = 100) to 512,955 bits (n = 3,125), i.e. 164 bits per comment at full size, while the "online"
accuracy of the pre-update predictions rises from 0.246 to 0.463 (means over 5 seeds). Because count
models are order-invariant, the final models — and so held-out predictions — are identical to the
batch baseline E03 (verified on all 40 runs).

**The class prior is the code for the label (E06, E15, E22).** argmin_e [L(x | M_e) − log₂ P(e)] is the
two-part code for the *pair* (label, text). It hurt macro-F1 everywhere: 4-class 0.495 → 0.447 at full
size (E03 vs E06), 27-class add-1 0.279 → 0.230 (E14 vs E15), 27-class KN validation 0.266 → 0.260
(E20 vs E22). Spending fewer bits on frequent labels is optimal for compressing the label sequence,
but macro-F1 rewards recall on rare labels, so "shortest joint code" and "best balanced classifier"
pull apart.

## 4. Bayes and MDL are the same rule

Because ℓ = −log₂ P, maximum-likelihood generative classification and conditional MDL are the same
decision rule. We implemented bigram Naive Bayes independently (own counting and scoring code) and
compared predictions: **0 of 2,984 test predictions differ** from conditional MDL at n = 500 and at
n = 23,485 (E28 vs E16), and **0 of 2,984** differ between MAP Naive Bayes and MDL with the prior code
(E29 vs E15). With base-2 logarithms the scores are bit-identical (unit tests and E28). The MDL
vocabulary adds nothing computationally — it adds the *interpretation*: the prior is a code for the
label, smoothing is a code for the unseen, and model cost is a code for the hypothesis.

## 5. Occam: context length and model complexity

Longer contexts are richer hypotheses; whether they pay depends on whether the data can support them.

| Track | Code | n = 2 | n = 3 | n = 4 |
|---|---|---|---|---|
| 4-class (val / test macro-F1) | add-1 | **0.574 / 0.606** (E09) | 0.433 / 0.495 (E03) | 0.371 / 0.455 (E10) |
| 27-class (val / test macro-F1) | add-1 | **0.229 / 0.279** (E16) | 0.204 / 0.270 (E23) | 0.178 / 0.231 (E24) |
| 27-class (val / test macro-F1) | Kneser-Ney | **0.266 / 0.304** (E20) | 0.263 / 0.294 (E25) | 0.263 / 0.293 (E26) |

(val = mean over all ladder sizes; test = full training set.) Under add-1, every extra word of context
fragments the counts further and the code degrades sharply. Kneser-Ney backs off to the shorter
context when the longer one is unseen, so longer contexts become **nearly free but not useful**: with
~800–900 short comments per class there is too little data for 3- and 4-word contexts to carry
information. The Occam lesson is empirical, not dogmatic: the best code is the simplest one the data
can actually pin down.

## 6. A neural bigram code (E30)

We also trained one neural bigram language model per class (embedding 128 → softmax over V = 18,727),
early-stopped on validation, and classified by summed bits.

* **As a code it sits between add-1 and KN.** Own-class validation code length, training-weighted:
  add-1 12.94 bits/token, neural 10.29, Kneser-Ney 8.73 (uniform: 14.19). The neural code beats add-1
  on 27/27 classes and loses to KN on 27/27.
* **As a classifier it is worse than both** (single seed, full size): macro-F1 0.220, accuracy 0.304.
  Conditional MDL classification relies on the 27 codes being *comparable*; independently trained,
  independently early-stopped networks (best epochs 5–35) are not calibrated against each other,
  while count codes with identical smoothing are.
* **Under two-part MDL it would be rejected outright.** The 27 networks have 129,950,109 parameters;
  at 32 bits each, L(M) ≈ 4.16·10⁹ bits, against L(D | M) ≈ 2.22·10⁶ bits for the 325,450 training
  tokens they model. A uniform code of the same tokens costs ≈ 4.6·10⁶ bits. The parameters cost
  ~1,900× more than the data they explain — exactly the situation MDL's model term exists to catch.

## 7. MDL vs LSTM: inductive bias and data efficiency

Test macro-F1 by training size (mean over 5 seeds):

| 4-class | 100 | 500 | 1,000 | 1,500 | 2,500 | 3,125 |
|---|---|---|---|---|---|---|
| MDL bigram add-1 (E09) | **0.362** | **0.506** | **0.538** | 0.569 | 0.596 | 0.606 |
| LSTM (E04) | 0.241 | 0.444 | 0.515 | **0.586** | **0.658** | **0.645** |

| 27-class | 500 | 2,000 | 4,000 | 12,000 | 16,000 | 23,485 |
|---|---|---|---|---|---|---|
| MDL Kneser-Ney (E20) | **0.166** | **0.233** | **0.258** | **0.288** | 0.297 | 0.304 |
| MDL add-1 (E16) | 0.149 | 0.205 | 0.231 | 0.259 | 0.268 | 0.279 |
| LSTM (E27) | 0.065 | 0.139 | 0.174 | 0.274 | **0.306** | **0.354** |

(Ladder sizes 250 and 1,500 are realised as 249 and 1,499 examples.) **The compression-based
classifier wins in the low-data regime on both tracks; the LSTM overtakes it** between 1,000 and 1,500
examples on the 4-class track (≈ 250–375 per class) and between 12,000 and 16,000 on the 27-class
track (≈ 440–590 per class). Reading: the n-gram code has a strong, nearly parameter-free inductive
bias ("local word statistics identify the class"), so it is data-efficient but saturates; the LSTM
must learn its representation from data — expensive at small n, but it can use long-range context and
is trained discriminatively for the decision itself. In MDL terms, at small n the simple hypothesis
class gives the shorter total description; with more data a richer hypothesis becomes affordable.

## 8. 4 classes vs 27 classes

Raw scores are not comparable across tracks (always predicting the majority class gives accuracy
0.335 on the 4-class test set and 0.117 on the 27-class test set), so we use κ and a shared subset of
texts. Full training set, κ: 4-class MDL 0.511 and LSTM 0.533; 27-class MDL Kneser-Ney 0.354, add-1
0.379, LSTM 0.421. At similar data per class (~780 vs ~870 examples), the 27-way problem is clearly
harder.

On the **same 391 test texts** (true label anger, fear, joy or sadness; seed 42, full size): 4-class
MDL 0.645 and LSTM 0.662 accuracy; 27-class MDL add-1 0.202, Kneser-Ney 0.340, LSTM 0.379. The 27-class
KN model sends these texts to *neighbouring* emotions — annoyance (42), approval (29), amusement (21),
disappointment (19) — distinctions the 4-class task removes by construction. Much of the gap is the
task (separating near-synonymous emotions from a few hundred examples each), not the code.

## 9. Limitations

* **K is uncomputable.** Every description length here is an upper bound relative to a small,
  hand-chosen model class (word n-grams, a neural bigram); conclusions are about these codes, not
  about Kolmogorov complexity itself.
* **Conditional, not full two-part MDL.** We do not encode the models; the one attempt to add model
  cost per text (E05) shows why that must be amortized, and prequential coding (E07) was only run on
  the 4-class track.
* **Single-label filtering.** Both single-label tracks discard multi-emotion comments (and neutral),
  i.e. exactly the ambiguous cases; results are optimistic about separability.
* **Small test counts for rare classes.** The 27-class test set has 2 grief, 7 pride, 7 relief and
  12 nervousness examples; their F1 values are extremely noisy, and macro-F1 inherits that noise.
* **Single runs where marked.** E28–E30 are seed-42 runs only (E28/E29 are exact-equivalence checks,
  so seeds do not matter for them; E30 is untuned).
