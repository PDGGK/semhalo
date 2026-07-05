#!/usr/bin/env python3
"""
Edge-Path re-ranking optimiser (the "semantic-halo" method).

Best configuration found empirically:
- path aggregation: mean (average BGE score over all tables on the path)
- length decay: linear (1/len), giving an overall 1/|p|^2 polynomial decay
- boost_factor: 0.4 * max_bge
- path contribution: accumulated (a table on many paths gets a higher score)

Core idea:
1. No 1-hop expansion -- only look at edges *between* the candidate tables.
2. Propagate semantic score along paths (a high-scoring table lends its
   "halo" to lower-scoring but structurally necessary bridge tables).
3. Tables on a path get a boost; shorter paths give a larger boost.

The relationship source is injectable: pass any callable
`get_relationships(table_names) -> list[(src, dst, rel_type, weight)]`.
This lets the optimiser run against a graph database in production, or
against an in-memory edge list in tests/demos with zero dependencies.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Set, Tuple

Relationship = Tuple[str, str, str, float]  # (source, target, rel_type, weight)
RelationshipFn = Callable[[Sequence[str]], List[Relationship]]


class EdgePathOptimizer:
    """Edge/path-based table re-ranking optimiser."""

    def __init__(
        self,
        get_relationships: RelationshipFn,
        boost_factor_mult: float = 0.4,   # boost = max_bge * 0.4
        max_path_length: int = 3,          # longest path (in nodes)
        use_edge_boost: bool = True,       # blend the auxiliary edge term
        edge_weight: float = 0.3,          # edge term weight (path gets 1 - edge_weight)
    ):
        self.get_relationships = get_relationships
        self.boost_factor_mult = boost_factor_mult
        self.max_path_length = max_path_length
        self.use_edge_boost = use_edge_boost
        self.edge_weight = edge_weight

    def optimize(
        self,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidate tables.

        Args:
            candidates: BGE retrieval result [(table_name, bge_score), ...]

        Returns:
            re-ranked tables [(table_name, final_score), ...]
        """
        if len(candidates) < 2:
            return candidates

        table_names = [t for t, _ in candidates]
        bge_scores = {t: s for t, s in candidates}

        # 1. edges among the candidate tables only (no neighbour expansion)
        relationships = self.get_relationships(table_names)
        if not relationships:
            return candidates  # no structure to exploit; keep BGE order

        # 2. adjacency
        adj: Dict[str, list] = defaultdict(list)
        for source, target, rel_type, weight in relationships:
            adj[source].append((target, rel_type, weight))
            adj[target].append((source, rel_type, weight))

        # 3. scores
        max_bge = max(bge_scores.values())
        if max_bge <= 0:
            return candidates  # no semantic signal to propagate
        boost_factor = max_bge * self.boost_factor_mult

        path_contributions = self._compute_path_contributions(
            table_names, adj, bge_scores, max_bge
        )
        if self.use_edge_boost:
            edge_contributions = self._compute_edge_contributions(
                table_names, adj, bge_scores, max_bge
            )
        else:
            edge_contributions = {t: 0.0 for t in table_names}

        # 4. fuse (boost-only: graph signal can only add)
        final_scores = {}
        for table in table_names:
            path_c = path_contributions.get(table, 0.0)
            edge_c = edge_contributions.get(table, 0.0)
            if self.use_edge_boost:
                combined = (1 - self.edge_weight) * path_c + self.edge_weight * edge_c
            else:
                combined = path_c
            final_scores[table] = bge_scores[table] + boost_factor * combined

        # 5. sort
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    def _compute_path_contributions(
        self,
        table_names: List[str],
        adj: Dict,
        bge_scores: Dict[str, float],
        max_bge: float,
    ) -> Dict[str, float]:
        """
        path_score(p) = (mean BGE over p) * (1/|p|) / max_bge   [= (1/(|p|^2 * s_max)) * sum s_u]
        Each table accumulates the score of every path it lies on; then max-normalise to [0, 1].
        """
        paths = self._find_all_paths(table_names, adj, self.max_path_length)

        path_contributions: Dict[str, float] = defaultdict(float)
        for path in paths:
            path_bge_avg = sum(bge_scores.get(t, 0.0) for t in path) / len(path)
            length_factor = 1.0 / len(path)
            path_score = path_bge_avg * length_factor / max_bge
            for table in path:
                path_contributions[table] += path_score

        if path_contributions:
            max_contrib = max(path_contributions.values())
            if max_contrib > 0:
                path_contributions = {
                    t: v / max_contrib for t, v in path_contributions.items()
                }
        return dict(path_contributions)

    def _compute_edge_contributions(
        self,
        table_names: List[str],
        adj: Dict,
        bge_scores: Dict[str, float],
        max_bge: float,
    ) -> Dict[str, float]:
        """Auxiliary term: a table connected by an edge gets weight * neighbour's normalised BGE."""
        edge_contributions: Dict[str, float] = {}
        table_set = set(table_names)
        for table in table_names:
            neighbours = adj.get(table, [])
            if not neighbours:
                edge_contributions[table] = 0.0
                continue
            edge_boost = 0.0
            for neighbour, _rel_type, weight in neighbours:
                if neighbour in table_set:
                    edge_boost += weight * (bge_scores.get(neighbour, 0.0) / max_bge)
            edge_contributions[table] = min(edge_boost, 1.0)

        if edge_contributions:
            max_contrib = max(edge_contributions.values())
            if max_contrib > 0:
                edge_contributions = {
                    t: v / max_contrib for t, v in edge_contributions.items()
                }
        return edge_contributions

    def _find_all_paths(
        self,
        table_names: List[str],
        adj: Dict,
        max_length: int = 3,
    ) -> List[List[str]]:
        """All simple paths of length 2..max_length among the candidate tables."""
        paths: List[List[str]] = []
        table_set = set(table_names)

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) >= 2:
                paths.append(path.copy())
            if len(path) >= max_length:
                return
            for neighbour, _, _ in adj.get(current, []):
                if neighbour in table_set and neighbour not in visited:
                    visited.add(neighbour)
                    path.append(neighbour)
                    dfs(neighbour, path, visited)
                    path.pop()
                    visited.remove(neighbour)

        for table in table_names:
            dfs(table, [table], {table})
        return paths
