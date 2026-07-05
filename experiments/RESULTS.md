# Spider schema-linking results (public benchmark, reproducible)

Run: `python spider_experiment.py --spider-dir <spider>` on Spider dev
(1034 questions, 166 databases). Lightweight IDF retriever, no LLM/API.
Metric: recall@k = fraction of questions whose entire gold table set is in top-k.

## All questions
| strategy    | R@1 | R@2 | R@3 | R@5 |
|-------------|-----|-----|-----|-----|
| table-only  | 53.1 | 80.5 | 93.6 | 98.5 |
| union       | 47.7 | 78.0 | 93.1 | 98.4 |
| union+edge  | 47.0 | 80.8 | **95.3** | **99.2** |

## Multi-table subset (378 questions — where join structure matters)
| strategy    | R@2 | R@3 | R@5 |
|-------------|-----|-----|-----|
| table-only  | 57.4 | 85.7 | 96.0 |
| union       | 51.9 | 84.7 | 96.0 |
| union+edge  | **59.3** | **89.9** | **97.9** |

## Reading
- **union + edge-path is best at R@3 and R@5**, the operating points that matter
  for schema linking (you feed the model the top-k tables). On the multi-table
  subset it adds +4.2pp at R@3 (85.7 -> 89.9) and +1.9pp at R@5.
- **Union alone does not help on Spider**: its schemas are small and dense, so the
  column channel mostly adds noise at low k. The method is designed for large,
  sparse schemas; here the *graph re-ranking* is what recovers and surpasses the
  baseline. This is honest and expected.
- Fully reproducible on public data; no company data, no model, no API.
