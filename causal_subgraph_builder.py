#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Temporal Subgraph Builder & Anti-Leakage Verifier for HyKGA
==================================================================
Repository: https://github.com/GUEDDES/HyKGA-Neuro-Symbolic-Graph-Transformers-withOntology-Guided-Attention

Implements Equations (18)-(20), Section 3.3.2-3.3.3, Section 3.5.1, and Algorithm 1 of the manuscript:
  E_i^{causal} = { (u, r, v, t_e) in E_hist |
                   t_e < tau_i  AND  {u, v} != {e_h^(i), e_t^(i)}  AND  (u, r, v, t_e) not in E_eval_targets }

Enforces three strict anti-leakage barriers across ALL structural inputs during training, validation, and testing:
  1. Strict Temporal Causality (t_e < tau_i; and t_e <= t_s < tau_i for centrality snapshots)
  2. Direct Target-Pair Excision ({u, v} != {e_h, e_t} across all relations and both directions)
  3. Split Isolation (held-out validation/test targets excluded from historical store E_hist)

Specifically verifies that target-pair exclusion is consistently applied to:
  (a) k-hop GAT neighborhood embeddings (Equations 18-19),
  (b) 4-scale causal snapshot centralities PE_graph(v, t_s; e_h, e_t) on G_{<= t_s}^{(-i)} (Equation 20), and
  (c) Causal Louvain-guided contrastive hard-negative sampling (Section 3.5.1, Equation 28).
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class TimestampedEdge:
    head: str
    relation: str
    tail: str
    timestamp_days: float  # timestamp in days (or Unix epoch converted to days)
    split: str = "train"   # 'train', 'val', or 'test'
    is_eval_target: bool = False
    topics: Tuple[str, ...] = field(default_factory=tuple)


class CausalTemporalGraphStore:
    """
    Stores historical social interactions and extracts strictly causal,
    target-pair-excised k-hop subgraphs, target-excised causal centrality snapshots,
    and causal Louvain community partitions for contrastive negative sampling.
    """

    def __init__(self, edges: Iterable[TimestampedEdge], gamma: float = 0.1):
        self.gamma = gamma
        all_edges = list(edges)
        # Barrier (iii): Held-out validation/test targets never enter E_hist
        self.edges: List[TimestampedEdge] = [
            e for e in all_edges if not e.is_eval_target and e.split == "train"
        ]
        self.eval_targets: Set[Tuple[str, str, str, float]] = {
            (e.head, e.relation, e.tail, e.timestamp_days)
            for e in all_edges
            if e.is_eval_target or e.split in ("val", "test")
        }
        # Cache of base periodic snapshots G_{<= t_s} (already free of E_eval_targets)
        self._snapshot_cache: Dict[float, Dict[str, Tuple[float, float, float, float]]] = {}
        self._snapshot_pair_index: Dict[float, Set[FrozenSet[str]]] = {}

    def temporal_decay(self, tau_i: float, t_e: float) -> float:
        """Implements exponential temporal decay alpha(tau_i, t_e) = exp(-gamma * (tau_i - t_e)) (Eq. 19)."""
        return math.exp(-self.gamma * max(0.0, tau_i - t_e))

    def build_causal_edge_set(
        self,
        tau_i: float,
        e_h: str,
        e_t: str,
    ) -> List[TimestampedEdge]:
        """
        Implements Equation (18):
          E_i^{causal} = { (u, r, v, t_e) in E_hist |
                           t_e < tau_i  and  {u, v} != {e_h, e_t}  and  (u, r, v, t_e) not in E_eval_targets }
        """
        target_pair = {e_h, e_t}
        causal_edges: List[TimestampedEdge] = []
        for e in self.edges:
            if e.timestamp_days >= tau_i:
                continue
            if {e.head, e.tail} == target_pair:
                continue
            if (e.head, e.relation, e.tail, e.timestamp_days) in self.eval_targets:
                continue
            causal_edges.append(e)
        return causal_edges

    def extract_khop_neighborhood(
        self,
        seed_entities: Iterable[str],
        tau_i: float,
        e_h: str,
        e_t: str,
        k: int = 2,
    ) -> Tuple[List[TimestampedEdge], Dict[str, Dict[str, float]]]:
        """
        Extracts the k-hop causal subgraph around seed_entities and computes normalized
        time-decay weights alpha_bar_v = alpha(tau_i, t_{(e,v)}) / sum_u alpha(tau_i, t_{(e,u)})
        (Equation 19).
        """
        causal_edges = self.build_causal_edge_set(tau_i=tau_i, e_h=e_h, e_t=e_t)
        adj: Dict[str, List[TimestampedEdge]] = defaultdict(list)
        for e in causal_edges:
            adj[e.head].append(e)
            adj[e.tail].append(e)

        visited: Set[str] = set(seed_entities)
        frontier: Set[str] = set(seed_entities)
        sub_edges: Set[TimestampedEdge] = set()

        for _ in range(k):
            next_frontier: Set[str] = set()
            for u in frontier:
                for e in adj.get(u, []):
                    sub_edges.add(e)
                    other = e.tail if e.head == u else e.head
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier

        # Compute normalized exponential time-decay weights per seed entity
        decay_weights: Dict[str, Dict[str, float]] = {}
        for seed in seed_entities:
            latest_ts: Dict[str, float] = {}
            for e in sub_edges:
                if e.head == seed or e.tail == seed:
                    nbr = e.tail if e.head == seed else e.head
                    if nbr not in latest_ts or e.timestamp_days > latest_ts[nbr]:
                        latest_ts[nbr] = e.timestamp_days
            raw = {
                nbr: math.exp(-self.gamma * (tau_i - ts))
                for nbr, ts in latest_ts.items()
            }
            total = sum(raw.values())
            decay_weights[seed] = (
                {nbr: w / total for nbr, w in raw.items()} if total > 0.0 else {}
            )

        return sorted(sub_edges, key=lambda x: x.timestamp_days), decay_weights

    @staticmethod
    def compute_causal_snapshot_centralities(
        edges: Iterable[TimestampedEdge],
        t_s: float,
        e_h: Optional[str] = None,
        e_t: Optional[str] = None,
        pagerank_damping: float = 0.85,
        pagerank_iters: int = 30,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Computes the 4-scale causal graph positional encoding inputs (Equation 20):
          [deg_{<= t_s}^{(-i)}(v), PR_{<= t_s}^{(-i)}(v), Betw_{<= t_s}^{(-i)}(v), Clos_{<= t_s}^{(-i)}(v)]
        over the target-pair-excised snapshot G_{<= t_s}^{(-i)} = G_{<= t_s} \\ E_pair(e_h, e_t),
        using strictly historical edges with t_e <= t_s < tau_i, excluding E_eval_targets
        AND excluding any direct edge between {e_h, e_t}.
        """
        target_pair: Optional[Set[str]] = {e_h, e_t} if (e_h is not None and e_t is not None) else None
        nodes: Set[str] = set()
        undirected: Dict[str, Set[str]] = defaultdict(set)
        directed_out: Dict[str, Set[str]] = defaultdict(set)

        for e in edges:
            if e.timestamp_days > t_s:
                continue
            if e.is_eval_target or e.split in ("val", "test"):
                continue
            if target_pair is not None and {e.head, e.tail} == target_pair:
                continue
            nodes.add(e.head)
            nodes.add(e.tail)
            if e.head != e.tail:
                undirected[e.head].add(e.tail)
                undirected[e.tail].add(e.head)
                directed_out[e.head].add(e.tail)

        n = len(nodes)
        if n == 0:
            return {}

        node_list = sorted(nodes)
        # 1. Normalized degree centrality
        deg = {u: len(undirected[u]) / max(n - 1, 1) for u in node_list}

        # 2. Power-iteration PageRank on directed graph
        pr = {u: 1.0 / n for u in node_list}
        for _ in range(pagerank_iters):
            next_pr = {u: (1.0 - pagerank_damping) / n for u in node_list}
            sink_mass = sum(pr[u] for u in node_list if len(directed_out[u]) == 0)
            for u in node_list:
                next_pr[u] += pagerank_damping * sink_mass / n
                out_neighbors = directed_out[u]
                if out_neighbors:
                    share = pagerank_damping * pr[u] / len(out_neighbors)
                    for v in out_neighbors:
                        next_pr[v] += share
            pr = next_pr

        # 3 & 4. Brandes betweenness and closeness centrality
        betw = {u: 0.0 for u in node_list}
        clos = {u: 0.0 for u in node_list}

        for s in node_list:
            stack: List[str] = []
            pred: Dict[str, List[str]] = {w: [] for w in node_list}
            sigma: Dict[str, float] = {w: 0.0 for w in node_list}
            sigma[s] = 1.0
            dist: Dict[str, int] = {w: -1 for w in node_list}
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in undirected[v]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            reachable = [d for u_node, d in dist.items() if u_node != s and d > 0]
            if reachable:
                clos[s] = (len(reachable) / max(n - 1, 1)) * (
                    len(reachable) / sum(reachable)
                )

            delta: Dict[str, float] = {w: 0.0 for w in node_list}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betw[w] += delta[w]

        norm_b = max((n - 1) * (n - 2), 1)
        for u in node_list:
            betw[u] /= norm_b

        return {u: (deg[u], pr[u], betw[u], clos[u]) for u in node_list}

    def get_leakage_free_snapshot_centralities(
        self,
        t_s: float,
        e_h: str,
        e_t: str,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Returns the 4-scale centrality dictionary for snapshot timestamp t_s < tau_i
        guaranteed to exclude all edges between (e_h, e_t) as well as all E_eval_targets.
        - If the cached base snapshot G_{<= t_s} contains no prior direct edge between
          e_h and e_t (which holds for all pairs without prior training interactions),
          the cached snapshot is already identical to G_{<= t_s}^{(-i)} and is returned in O(1).
        - If a prior historical edge between {e_h, e_t} exists at t_e <= t_s, centralities
          are computed on the target-pair-excised snapshot G_{<= t_s} \\ E_pair(e_h, e_t).
        """
        if t_s not in self._snapshot_cache:
            self._snapshot_cache[t_s] = self.compute_causal_snapshot_centralities(
                self.edges, t_s=t_s, e_h=None, e_t=None
            )
            self._snapshot_pair_index[t_s] = {
                frozenset((e.head, e.tail))
                for e in self.edges
                if e.timestamp_days <= t_s and e.head != e.tail
            }

        pair_key = frozenset((e_h, e_t))
        if pair_key not in self._snapshot_pair_index[t_s]:
            return self._snapshot_cache[t_s]

        # Prior direct edge existed before t_s -> recompute on target-excised G_{<= t_s}^{(-i)}
        return self.compute_causal_snapshot_centralities(
            self.edges, t_s=t_s, e_h=e_h, e_t=e_t
        )

    def compute_causal_communities_and_negatives(
        self,
        tau_i: float,
        e_h: str,
        e_t: str,
        query_topics: Set[str],
    ) -> Tuple[Dict[str, int], List[TimestampedEdge]]:
        """
        Implements Causal Louvain-Guided Negative Sampling (Section 3.5.1):
          1. Constructs E_i^{causal} with t_e < tau_i, {u, v} != {e_h, e_t}, and E_eval_targets excluded.
          2. Partitions nodes into modularity communities on E_i^{causal} (greedy modularity / label propagation).
          3. Selects hard-negative edges/subgraphs from E_i^{causal} that share topics/hashtags with the
             query message but belong to a distinct topological community from e_h and e_t.
        """
        causal_edges = self.build_causal_edge_set(tau_i=tau_i, e_h=e_h, e_t=e_t)
        adj: Dict[str, Set[str]] = defaultdict(set)
        nodes: Set[str] = {e_h, e_t}
        for e in causal_edges:
            nodes.add(e.head)
            nodes.add(e.tail)
            if e.head != e.tail:
                adj[e.head].add(e.tail)
                adj[e.tail].add(e.head)

        # Deterministic modularity-based community assignment on E_i^{causal}
        sorted_nodes = sorted(nodes)
        comm: Dict[str, int] = {u: idx for idx, u in enumerate(sorted_nodes)}
        for _ in range(10):
            changed = False
            for u in sorted_nodes:
                if not adj[u]:
                    continue
                counts: Dict[int, int] = defaultdict(int)
                for nbr in adj[u]:
                    counts[comm[nbr]] += 1
                best_c = min(
                    counts.keys(),
                    key=lambda c: (-counts[c], c),
                )
                if best_c != comm[u]:
                    comm[u] = best_c
                    changed = True
            if not changed:
                break

        target_comms = {comm[e_h], comm[e_t]}
        hard_negatives: List[TimestampedEdge] = []
        for e in causal_edges:
            shares_topic = bool(query_topics.intersection(e.topics)) if query_topics else True
            in_different_comm = (comm[e.head] not in target_comms) and (comm[e.tail] not in target_comms)
            if shares_topic and in_different_comm:
                hard_negatives.append(e)

        return comm, hard_negatives


def run_self_test():
    """
    Verifies that target-pair edges, reverse edges, held-out evaluation targets, and future edges
    are strictly excised across:
      1. k-hop GAT neighborhood extraction (Eqs. 18-19)
      2. 4-scale causal centrality snapshots PE_graph (Eq. 20)
      3. Causal Louvain contrastive negative sampling (Section 3.5.1)
    """
    sample_edges = [
        # Direct prior edges between target pair (u1, u2) that MUST be excised when predicting (u1, u2)
        TimestampedEdge("u1", "follows", "u2", timestamp_days=1.0, split="train", topics=("#AI",)),
        TimestampedEdge("u2", "interacts_with", "u1", timestamp_days=2.0, split="train", topics=("#AI",)),
        # Indirect 2-hop path u1 -> u3 -> u2 (admissible historical context)
        TimestampedEdge("u1", "follows", "u3", timestamp_days=3.0, split="train", topics=("#AI",)),
        TimestampedEdge("u3", "follows", "u2", timestamp_days=4.0, split="train", topics=("#Tech",)),
        # Separate community edge sharing topic #AI (for Louvain hard negative sampling)
        TimestampedEdge("u5", "follows", "u6", timestamp_days=3.5, split="train", topics=("#AI",)),
        # Future edge relative to tau_i = 5.0 (MUST be excised)
        TimestampedEdge("u3", "mentions", "u2", timestamp_days=6.0, split="train", topics=("#AI",)),
        # Held-out test evaluation target (MUST never enter E_hist or any snapshot)
        TimestampedEdge("u1", "follows", "u4", timestamp_days=2.5, split="test", is_eval_target=True),
    ]
    store = CausalTemporalGraphStore(sample_edges, gamma=0.1)

    # 1. Verify k-hop neighborhood anti-leakage
    causal_sub, weights = store.extract_khop_neighborhood(
        seed_entities=["u1", "u2"], tau_i=5.0, e_h="u1", e_t="u2", k=2
    )
    for e in causal_sub:
        assert e.timestamp_days < 5.0, "Future edge leaked into E_i^{causal}!"
        assert {e.head, e.tail} != {"u1", "u2"}, "Direct target-pair edge leaked into E_i^{causal}!"
        assert not e.is_eval_target, "Held-out evaluation target leaked into E_i^{causal}!"

    # 2. Verify Centrality Snapshot anti-leakage (Equation 20)
    centralities_excised = store.get_leakage_free_snapshot_centralities(t_s=4.5, e_h="u1", e_t="u2")
    centralities_unfiltered = store.compute_causal_snapshot_centralities(
        sample_edges, t_s=4.5, e_h=None, e_t=None
    )
    # In the excised snapshot G_{<= 4.5}^{(-(u1,u2))}, u1 only connects to u3 (1 neighbor among 5 active nodes -> deg = 1/4 = 0.25),
    # whereas without target-pair excision u1 would also connect to u2 (deg = 2/4 = 0.50).
    assert abs(centralities_excised["u1"][0] - 0.25) < 1e-9, (
        f"Target edge leaked into degree centrality: got {centralities_excised['u1'][0]}, expected 0.25"
    )
    assert centralities_excised["u1"][0] < centralities_unfiltered["u1"][0], (
        "Excised centrality did not remove prior direct target-pair edge!"
    )
    assert "u4" not in centralities_excised, "Held-out evaluation target node u4 leaked into snapshot!"

    # 3. Verify Causal Louvain Contrastive Negative Sampling anti-leakage
    comm, hard_negs = store.compute_causal_communities_and_negatives(
        tau_i=5.0, e_h="u1", e_t="u2", query_topics={"#AI"}
    )
    for e in hard_negs:
        assert e.timestamp_days < 5.0, "Future edge leaked into contrastive negatives!"
        assert {e.head, e.tail} != {"u1", "u2"}, "Target pair leaked into contrastive negatives!"
    assert len(hard_negs) == 1 and {hard_negs[0].head, hard_negs[0].tail} == {"u5", "u6"}

    print("=== Causal Subgraph Builder & Anti-Leakage Self-Test ===")
    print(f"[1] k-hop GAT subgraph for (u1, u2) at tau_i=5.0: {len(causal_sub)} edges (direct (u1,u2) & future edges excised)")
    print(f"    Normalized time-decay weight u1 -> u3: {weights['u1'].get('u3', 0.0):.3f}")
    print(f"[2] Snapshot centralities at t_s=4.5 with target-pair excision G_{{<=t_s}}^{{(-i)}}:")
    print(f"    deg(u1) unexcised = {centralities_unfiltered['u1'][0]:.3f} -> deg(u1) target-excised = {centralities_excised['u1'][0]:.3f} (PASS)")
    print(f"[3] Causal Louvain hard negatives for (u1, u2) on topic '#AI': {len(hard_negs)} edge(s) from disjoint community (u5, u6) (PASS)")
    print("All 3 structural channels (GAT neighborhood, PE_graph centralities, Louvain sampling) verified leakage-free -> PASS")


if __name__ == "__main__":
    run_self_test()
