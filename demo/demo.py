#!/usr/bin/env python3
"""
Runnable demo -- no database, no embedding model, standard library only.

Scenario (fully synthetic, e-commerce):
    query ~ "how many products has each customer bought this month"

The `order` table is the one you must JOIN through
(customer -> order -> order_item -> product), but a keyword/semantic retriever
scores it LOW because the question never says "order". It is connected by
foreign keys to the top-scoring tables, so the edge-path re-ranker lifts it via
the "semantic halo" effect.

Run:  python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semhalo import EdgePathOptimizer  # noqa: E402

# Candidate tables from the (simulated) retriever: (table, score)
candidates = [
    ("product", 5.07),        # matches "products" -> ranked #1
    ("order_monthly", 2.77),
    ("customer", 2.53),
    ("v_order_today", 2.37),
    ("v_order_history", 2.35),
    ("order_daily", 2.16),
    ("order_total", 1.88),
    ("order_item", 1.72),
    ("category", 1.60),
    ("order", 1.55),          # the JOIN bridge -- ranked LAST
]

# Declared foreign-key edges (undirected)
fk_edges = [
    ("customer", "order"),
    ("order", "order_item"),
    ("order_item", "product"),
    ("product", "category"),
    ("order", "order_monthly"),
    ("order", "order_daily"),
    ("order", "order_total"),
    ("order", "v_order_today"),
    ("order", "v_order_history"),
]


def get_relationships(table_names):
    """Return FK edges that lie between the given candidate tables."""
    s = set(table_names)
    return [(a, b, "IS", 1.0) for a, b in fk_edges if a in s and b in s]


def show(title, ranking):
    print(f"\n{title}")
    print("-" * 42)
    for i, (table, score) in enumerate(ranking, 1):
        star = "  <-- JOIN bridge" if table == "order" else ""
        print(f"  {i:2d}. {table:20s} {score:6.3f}{star}")


def rank_of(ranking, table):
    return next(i for i, (t, _) in enumerate(ranking, 1) if t == table)


if __name__ == "__main__":
    reranked = EdgePathOptimizer(get_relationships).optimize(candidates)
    show("Retriever ranking", candidates)
    show("Edge-Path re-ranking", reranked)
    before, after = rank_of(candidates, "order"), rank_of(reranked, "order")
    print(f"\nBridge table 'order': #{before} -> #{after}  (+{before - after})")
    print("The structurally necessary table was lifted into the top ranks,")
    print("without demoting any table whose score was already clearly higher.")
