"""Seeded random maps and independent shortest safe-path distances."""

from collections import deque

import numpy as np


def distances_to_goal(rows):
    height, width = len(rows), len(rows[0])
    goal = "".join(rows).index("G")
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        state = queue.popleft()
        row, col = divmod(state, width)
        for r, c in ((row, col - 1), (row + 1, col), (row, col + 1), (row - 1, col)):
            nxt = r * width + c
            if (
                0 <= r < height
                and 0 <= c < width
                and rows[r][c] != "H"
                and nxt not in distances
            ):
                distances[nxt] = distances[state] + 1
                queue.append(nxt)
    return distances


def random_map(rng, size, safe_probability=0.75):
    cells = rng.choice(
        ["F", "H"], size=size * size, p=[safe_probability, 1 - safe_probability]
    )
    start, goal = rng.choice(size * size, size=2, replace=False)
    cells[start], cells[goal] = "S", "G"
    return ["".join(row) for row in cells.reshape(size, size)]


def generate_maps(count, seed, sizes=(8, 12), bins=(1, 2, 4, 8)):
    rng = np.random.default_rng(seed)
    seen = set()
    accepted = 0
    for _ in range(count * 1000):
        size = sizes[accepted % len(sizes)]
        rows = random_map(rng, size)
        key = tuple(rows)
        distances = distances_to_goal(rows)
        start = "".join(rows).index("S")
        if key in seen or start not in distances:
            continue
        groups = [
            [s for s, d in distances.items() if d == distance] for distance in bins
        ]
        if any(not group for group in groups):
            continue
        seen.add(key)
        states = [int(rng.choice(group)) for group in groups]
        yield rows, states, distances
        accepted += 1
        if accepted == count:
            return
    raise RuntimeError("Could not generate enough distinct maps with all distance bins")
