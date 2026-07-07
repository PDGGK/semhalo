#!/usr/bin/env python3
"""
Schema-linking retrieval experiment on the public Spider dev set.

Compares three strategies at recovering the gold tables of a question, using an
identical lightweight IDF retriever underneath (no external model, no LLM, no
API -- fully reproducible):

    1. table-only        : rank tables by table-text similarity (the baseline)
    2. union             : rank by max(table channel, column channel)
    3. union + edge-path : re-rank the union candidates with the FK graph

Metric: recall@k = fraction of questions whose *entire* gold table set is inside
the top-k. Reported for all questions and for the multi-table subset (where join
structure -- and therefore the graph -- actually matters).

Usage:
    python spider_experiment.py --spider-dir /path/to/spider              # aggregate numbers
    python spider_experiment.py --spider-dir /path/to/spider --examples 3 # + concrete cases
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semhalo import EdgePathOptimizer  # noqa: E402


def tokenize(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)          # camelCase -> two tokens
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(t) > 1]


def build_db(schema):
    """Return per-table token docs, per-column (table_idx, tokens), and table-level FK edges."""
    tables = schema["table_names_original"]
    cols = schema["column_names_original"]              # [[table_idx, col_name], ...]; index 0 is '*'
    table_tokens = [set(tokenize(t)) for t in tables]
    col_docs = []                                        # (table_idx, token_set)
    for tbl_idx, col_name in cols:
        if tbl_idx < 0:
            continue
        table_tokens[tbl_idx] |= set(tokenize(col_name))
        col_docs.append((tbl_idx, set(tokenize(col_name))))
    # foreign keys are column-index pairs -> table-index edges
    edges = set()
    for a, b in schema["foreign_keys"]:
        ta, tb = cols[a][0], cols[b][0]
        if ta != tb and ta >= 0 and tb >= 0:
            edges.add((min(ta, tb), max(ta, tb)))
    return tables, table_tokens, col_docs, edges


def idf_map(list_of_token_sets):
    n = len(list_of_token_sets)
    df = defaultdict(int)
    for ts in list_of_token_sets:
        for tok in ts:
            df[tok] += 1
    return {tok: math.log((n + 1) / (c + 0.5)) for tok, c in df.items()}, n


def score(query_tokens, doc_tokens, idf):
    return sum(idf.get(tok, 0.0) for tok in query_tokens if tok in doc_tokens)


def gold_tables(ex):
    return {tu[1] for tu in ex["sql"]["from"]["table_units"] if tu[0] == "table_unit"}


def recall_at_k(ranking, gold, k):
    return 1.0 if gold.issubset(set(ranking[:k])) else 0.0


def rank_one(question, schema, idf, topn):
    """Rank a DB's tables for one question. Returns (table_names, rank_table, rank_edge)."""
    tables, table_tokens, col_docs, edges = build_db(schema)
    n = len(tables)
    q = set(tokenize(question))

    tbl_score = {i: score(q, table_tokens[i], idf) for i in range(n)}
    col_best = defaultdict(float)
    for tbl_idx, ctoks in col_docs:
        col_best[tbl_idx] = max(col_best[tbl_idx], score(q, ctoks, idf))

    def norm(d):
        m = max(d.values()) if d and max(d.values()) > 0 else 1.0
        return {i: d.get(i, 0.0) / m for i in range(n)}
    nt, nc = norm(tbl_score), norm(col_best)
    union_score = {i: max(nt[i], nc[i]) for i in range(n)}

    rank_table = sorted(range(n), key=lambda i: tbl_score[i], reverse=True)
    rank_union = sorted(range(n), key=lambda i: union_score[i], reverse=True)

    pool = rank_union[:topn]
    cand = [(str(i), union_score[i]) for i in pool]
    edge_set = {(min(a, b), max(a, b)) for a, b in edges}

    def get_rel(names):
        s = {int(x) for x in names}
        return [(str(a), str(b), "IS", 1.0) for (a, b) in edge_set if a in s and b in s]

    reranked = EdgePathOptimizer(get_rel).optimize(cand)
    rank_edge = [int(t) for t, _ in reranked] + [i for i in rank_union if i not in pool]
    return tables, rank_table, rank_union, rank_edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-dir", required=True)
    ap.add_argument("--topn", type=int, default=15, help="candidate pool for edge-path")
    ap.add_argument("--examples", type=int, default=0,
                    help="also print N concrete cases where edge-path recovers a bridge table the baseline missed")
    args = ap.parse_args()

    tables_json = json.load(open(os.path.join(args.spider_dir, "tables.json")))
    dev = json.load(open(os.path.join(args.spider_dir, "dev.json")))
    schemas = {s["db_id"]: s for s in tables_json}

    # global IDF over every table-doc across all DBs (more robust than per-DB)
    all_table_docs = []
    for s in tables_json:
        _, tt, _, _ = build_db(s)
        all_table_docs.extend(tt)
    idf, _ = idf_map(all_table_docs)

    ks = [1, 2, 3, 5]
    agg = {s: {k: [] for k in ks} for s in ("table", "union", "edgepath")}
    agg_multi = {s: {k: [] for k in ks} for s in ("table", "union", "edgepath")}
    wins = []  # concrete cases where union+edge uniquely covers all gold @3

    for ex in dev:
        schema = schemas[ex["db_id"]]
        gold = gold_tables(ex)
        if not gold:
            continue
        names, rank_table, rank_union, rank_edge = rank_one(ex["question"], schema, idf, args.topn)

        rankings = {"table": rank_table, "union": rank_union, "edgepath": rank_edge}
        for strat, r in rankings.items():
            for k in ks:
                v = recall_at_k(r, gold, k)
                agg[strat][k].append(v)
                if len(gold) >= 2:
                    agg_multi[strat][k].append(v)

        # a concrete win: multi-table, edge-path gets all gold into top-3, baseline does not
        if args.examples and len(gold) >= 2 and \
                recall_at_k(rank_edge, gold, 3) == 1 and recall_at_k(rank_table, gold, 3) == 0:
            pos_t = {i: r for r, i in enumerate(rank_table, 1)}
            pos_e = {i: r for r, i in enumerate(rank_edge, 1)}
            lifted = max(gold, key=lambda g: pos_t[g] - pos_e[g])  # biggest rank gain
            wins.append({
                "db": ex["db_id"], "q": ex["question"],
                "gold": sorted(names[i] for i in gold),
                "bridge": names[lifted], "before": pos_t[lifted], "after": pos_e[lifted],
                "top_table": [names[i] for i in rank_table[:5]],
                "top_edge": [names[i] for i in rank_edge[:5]],
            })

    def report(name, table):
        print(f"\n{name}")
        print(f"{'strategy':<12} " + "  ".join(f"R@{k}" for k in ks))
        for strat in ("table", "union", "edgepath"):
            row = "  ".join(f"{100*sum(table[strat][k])/len(table[strat][k]):5.1f}" for k in ks)
            label = {"table": "table-only", "union": "union", "edgepath": "union+edge"}[strat]
            print(f"{label:<12} {row}")

    print(f"Spider dev: {len(dev)} questions, {len(schemas)} databases")
    report("Recall@k (all questions):", agg)
    report(f"Recall@k (multi-table subset, n={len(agg_multi['table'][1])}):", agg_multi)

    if args.examples and wins:
        print(f"\n\nConcrete cases -- edge-path recovers a join-bridge table the plain retriever "
              f"drops (showing {min(args.examples, len(wins))} of {len(wins)} such cases):")
        for w in wins[:args.examples]:
            print(f"\n  db: {w['db']}   |   \"{w['q']}\"")
            print(f"  gold tables needed: {w['gold']}")
            print(f"  bridge table '{w['bridge']}':  #{w['before']} (table-only)  ->  #{w['after']} (union+edge)")
            print(f"    table-only top-5: {w['top_table']}")
            print(f"    union+edge top-5: {w['top_edge']}")


if __name__ == "__main__":
    main()
