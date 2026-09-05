"""Near-duplicate detection via perceptual hashing (D-011).

Why not exact file hashing: in a scanned corpus the same physical document rescanned
differs in noise, skew and JPEG quantization -- different bytes, identical content.
Exact hashing catches essentially nothing.

Why BEFORE splitting: running it after would tell us leakage happened without letting
us prevent it. The whole point is to keep near-duplicates inside one split.

Why union-find: near-duplicate relations are NOT transitive. A~B and B~C does not give
A~C. Connected components may over-merge slightly, which is the safe direction for
leakage prevention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnionFind:
    """Standard disjoint-set with path compression and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


@dataclass
class DedupReport:
    clusters: list[list[int]]          # only clusters of size > 1
    cross_label: list[list[int]]       # clusters spanning >1 class label
    n_images: int
    n_duplicated: int                  # images sitting in a multi-image cluster
    threshold: int


def hashes_to_bits(hashes) -> np.ndarray:
    """Stack imagehash objects into an (N, 64) uint8 bit matrix."""
    return np.stack([h.hash.flatten().astype(np.uint8) for h in hashes])


def pairwise_hamming(bits: np.ndarray) -> np.ndarray:
    """Full (N, N) Hamming distance matrix.

    Done as a matrix product rather than a Python loop: for N=3482 that is ~6M
    pairs, which is seconds in numpy and minutes in a loop.

    For 0/1 vectors, hamming(a, b) = popcount(a) + popcount(b) - 2 * (a . b).
    """
    b = bits.astype(np.int32)
    pop = b.sum(axis=1)
    dot = b @ b.T
    return pop[:, None] + pop[None, :] - 2 * dot


def find_clusters(hashes, labels: np.ndarray, threshold: int = 5) -> DedupReport:
    """Group images whose perceptual hashes are within `threshold` bits."""
    bits = hashes_to_bits(hashes)
    n = len(bits)
    dist = pairwise_hamming(bits)

    uf = UnionFind(n)
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if dist[i, j] <= threshold:
            uf.union(int(i), int(j))

    comps = [sorted(c) for c in uf.components().values() if len(c) > 1]
    comps.sort(key=len, reverse=True)

    cross = [c for c in comps if len({int(labels[i]) for i in c}) > 1]

    return DedupReport(
        clusters=comps,
        cross_label=cross,
        n_images=n,
        n_duplicated=sum(len(c) for c in comps),
        threshold=threshold,
    )


def cluster_ids(report: DedupReport, n: int) -> np.ndarray:
    """Map each image index -> a group id, for use as the split grouping key.

    Singletons get their own id. Passing this to a grouped split keeps every
    near-duplicate cluster inside a single split, which is the actual goal.
    """
    gid = np.arange(n)
    for c in report.clusters:
        for i in c:
            gid[i] = c[0]
    return gid
