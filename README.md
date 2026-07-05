# SemHalo — a "semantic-halo" edge-path re-ranker for schema linking

A small prototype exploring one idea for **schema linking** in Text-to-SQL: when a
database has many tables, a semantic retriever surfaces the obviously-relevant
ones but misses **bridge tables** — tables a join needs but that match the query
poorly on surface similarity.

The idea: let a high-scoring table lend some of its score to the low-scoring
tables it connects to through foreign keys — a "semantic halo" that flows along
short paths in the FK graph. Score is propagated along paths of length <= 3 with
a polynomial `1/|p|^2` length decay and a **boost-only** rule (structure can only
raise a score, never lower it), so it never demotes a table whose semantic score
was already clearly higher.

I tried standard Personalized PageRank first; on sparse FK graphs its 1-hop
expansion floods the candidate set and dilutes the signal, which is what led to
this local, path-based alternative.

## Run the demo (standard library only — no database, no model)

```bash
python demo/demo.py
```

A tiny synthetic sparse-FK schema where a low-scored bridge table is a join
necessity; the demo shows it lifted into the top ranks by the halo effect.

## A quick check on public data

`python experiments/spider_experiment.py --spider-dir <spider>` runs a small,
reproducible check on the Spider dev set (same IDF retriever for every strategy —
no LLM). Early result: on multi-table questions the graph re-ranking improves
recall@3 over a table-only baseline. A rough sanity check, not a proper study.

## Open questions (what I'd like to work on)

This is an early prototype; the interesting work is still ahead:

1. **Make it principled.** The edge-path rule is hand-tuned — can it be recast as
   a proper *query-guided* propagation with provable bounds instead of a heuristic?
2. **Scale it.** An efficient push-based backend for large, weight-skewed sparse
   graphs would let this run on much larger schemas.
3. **Evaluate it properly.** The Spider check is on small, dense schemas; the
   method targets *large, sparse* ones and needs a real study on larger-schema
   benchmarks and end-to-end SQL accuracy, not just retrieval recall.
4. **Characterise when it helps.** Which graph regimes make local path
   propagation beat global PPR?

Feedback and pointers welcome.
