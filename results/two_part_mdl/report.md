# Two-part MDL model selection (4-class, in-sample)

Training data D: 3125 comments (full 4-class training set). B = 32 bits per stored count entry.
L(D|M) is measured on the same comments the models were fitted on (in-sample),
so it is a dataset-description length, not a held-out estimate.

| config | count entries | L(M) bits | L(D|M) bits | L_total bits | L(M) share | bits/comment | test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bigram_add1 | 29,905 | 957,088 | 453,201 | 1,410,289 | 67.9% | 145.0 | 0.6063 |
| trigram_add1 | 37,298 | 1,193,664 | 469,998 | 1,663,662 | 71.7% | 150.4 | 0.4948 |
| bigram_kn **(selected)** | 29,905 | 957,216 | 221,159 | 1,178,375 | 81.2% | 70.8 | 0.7242 |
| uniform code (reference, not a candidate) | 0 | 0 | 523,703 | 523,703 | 0% | 167.6 | – |

Selected: **bigram_kn** — test macro-F1 0.7242, accuracy 0.7289.

Confusion matrix (rows true, columns predicted): anger, fear, joy, sadness

       anger    90     7     8    26
        fear     9    43     2    11
         joy     6     6    73     8
     sadness     6    11     6    79

## Ladder robustness (selections over 5 seeds per size)

|   size |   bigram_kn |
|-------:|------------:|
|    100 |           5 |
|    250 |           5 |
|    500 |           5 |
|   1000 |           5 |
|   1500 |           5 |
|   2000 |           5 |
|   2500 |           5 |
|   3125 |           5 |
