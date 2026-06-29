# capture-content-eval results

Frontier models label capture metadata from a text description: a primary type (subject vs scene) plus secondary labels with confidence. Scored on primary accuracy, label precision / recall / F1, and confidence calibration (Brier, lower is better).

## Scores

| model | primary acc | precision | recall | F1 | Brier |
|---|---|---|---|---|---|
| gpt | 95% | 96% | 95% | 95% | 0.042 |
| claude | 100% | 84% | 99% | 90% | 0.127 |

## Failure profile

| failure type | gpt | claude |
|---|---|---|
| wrong_primary | 1 | 0 |
| misses | 3 | 1 |
| over_claims | 3 | 11 |
| confident_over_claims | 3 | 9 |

## Takeaway

Both models score high (F1 95% for gpt, 90% for claude), but trade off differently: gpt's main weakness is 3 over-claiming, 3 of them confident (Brier 0.042); claude's main weakness is 11 over-claiming, 9 of them confident (Brier 0.127). Lower Brier means confidence you can trust unattended.
