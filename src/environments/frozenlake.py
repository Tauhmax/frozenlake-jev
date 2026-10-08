"""Small deterministic Gymnasium adapter; P is the source of truth."""

import gymnasium as gym


def make_env(rows):
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Map must be a nonempty rectangle")
    cells = "".join(rows)
    if set(cells) - set("SFHG") or cells.count("S") != 1 or cells.count("G") != 1:
        raise ValueError("Map needs exactly one S and G, with only S/F/H/G cells")
    return gym.make("FrozenLake-v1", desc=rows, is_slippery=False).unwrapped


def render_state(env, state):
    cells = env.desc.astype("U1").copy()
    if state not in range(env.observation_space.n) or cells.flat[state] in "HG":
        raise ValueError("Player must occupy a nonterminal safe state")
    cells[cells == "S"] = "F"
    cells.flat[state] = "P"
    return "\n".join("".join(row) for row in cells)
