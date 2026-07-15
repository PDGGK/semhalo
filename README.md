# SemHalo — semantic-halo edge-path re-ranking for schema linking

A small method for **schema linking** in Text-to-SQL: recovering the tables a
question needs — including the **bridge tables** a join requires but that a
retriever scores poorly on surface similarity.

## On real data (Spider dev, fully reproducible)

The same lightweight IDF retriever is used for every strategy — no LLM, no
embedding model, no API — so the comparison is reproducible from the repo alone.
Metric: recall@k = fraction of questions whose *entire* gold table set is in the
top-k.

**Multi-table questions** (378 of 1,034 — where join structure, and the graph,
actually matter):

| strategy               | R@2      | R@3      | R@5      |
| ---------------------- | -------- | -------- | -------- |
| table-only (baseline)  | 57.4     | 85.7     | 96.0     |
| **union + edge-path**  | **59.3** | **89.9** | **97.9** |

Beyond the aggregate, edge-path recovers a join-bridge table in **23 cases the
plain retriever drops out of the top-3**. A real example from the `car_1` DB:

> *"Find the model of the car whose weight is below the average weight."*
> needs `car_names` **and** `cars_data`. A plain retriever ranks `cars_data`
> only #4 (outside top-3); edge-path propagates score along the foreign-key path
> and lifts it to #3, recovering the full join.

```bash
python experiments/spider_experiment.py --spider-dir <spider> --examples 3
```

(The gains are modest — Spider's schemas are small and dense. The method targets
*large, sparse* schemas; see Open Questions.)

## The idea

When a database has many tables, a retriever surfaces the obviously-relevant
ones but misses **bridge tables** — needed by a join, scored low on their own.
The idea: let a high-scoring table lend some of its score to the low-scoring
tables it connects to through foreign keys — a "semantic halo" flowing along
short paths in the FK graph.

Score is propagated along paths of length ≤ 3 with a polynomial `1/|p|^2` length
decay and a **boost-only** rule (structure can only raise a score, never lower
it), so it never demotes a table whose score was already clearly higher. I tried
standard Personalized PageRank first; on sparse FK graphs its 1-hop expansion
floods the candidate set and washes out the signal, which led to this local,
path-based alternative.

## Layout

```
src/semhalo/edge_path.py           the edge-path optimiser (relationship source is injectable)
experiments/spider_experiment.py   the reproducible Spider evaluation above
demo/demo.py                       a zero-dependency synthetic illustration of the mechanism
```

Run `python demo/demo.py` for a tiny, no-data illustration (a synthetic sparse-FK
schema where a low-scored bridge table is lifted into the top ranks).

## Open questions (what I'd like to develop)

An early write-up with worked-out properties (order-preservation, scale-invariance,
a path-kernel form) exists; the interesting work is still ahead:

1. **Provable propagation bounds.** The properties above are about the fusion
   rule, not the propagation itself — can it be recast as a *query-guided*
   propagation with convergence / approximation guarantees?
2. **Scale.** An efficient push-based backend for large, weight-skewed sparse
   graphs would let this run on much larger schemas.
3. **Evaluate properly.** Spider's schemas are small and dense; the method
   targets large, sparse ones, and needs a study on larger-schema benchmarks and
   on end-to-end SQL accuracy, not just retrieval recall.

Feedback and pointers welcome.

## Authorship & status

© 2026 Zihan Dai. This repository documents original, ongoing research on
edge-path re-ranking for schema linking, first published here on 6 July 2026,
and now continued as a supervised research project at the University of
Melbourne. **All rights reserved** — this is not open-source software. If you
wish to build on this work, please get in touch: dai.z1@student.unimelb.edu.au
