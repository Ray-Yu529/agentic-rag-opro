"""
pareto.py — 多目標 Pareto 前緣 (server.py 與 plot.py 共用，避免兩份實作漂移)
"""

from __future__ import annotations


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points = (hallucination, correctness)。前緣 = 幻覺率低且正確率高。
    回傳依幻覺率遞增排序的前緣點。"""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best_corr = [], -1.0
    for hall, corr in pts:
        if corr > best_corr:
            front.append((hall, corr))
            best_corr = corr
    return front
