"""Discounted value iteration with terminal transitions masked."""

import numpy as np


def value_iteration(env, gamma=0.99, tolerance=1e-12, max_iterations=10000):
    if not 0 <= gamma < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require 0 <= gamma < 1 and finite positive tolerance")
    values = np.zeros(env.observation_space.n)

    def backup(v):
        return np.array(
            [
                [
                    sum(
                        p * (r + gamma * v[next_s] * (not terminal))
                        for p, next_s, r, terminal in env.P[s][a]
                    )
                    for a in range(env.action_space.n)
                ]
                for s in range(env.observation_space.n)
            ]
        )

    for iteration in range(1, max_iterations + 1):
        values = backup(values).max(axis=1)
        q_values = backup(values)
        residual = float(np.max(np.abs(q_values.max(axis=1) - values)))
        if residual <= tolerance:
            return q_values, {"iterations": iteration, "bellman_residual": residual}
    raise RuntimeError("Value iteration did not converge")


def optimal_actions(q_values, tolerance=1e-9):
    return np.flatnonzero(np.abs(q_values - np.max(q_values)) <= tolerance).tolist()


def finite_horizon_q(env, max_steps=30, gamma=0.99):
    """Q[h, state, action] with h moves left; horizon zero has no reward."""
    if max_steps < 1 or not 0 <= gamma <= 1:
        raise ValueError("Require positive horizon and 0 <= gamma <= 1")
    result = np.zeros((max_steps + 1, env.observation_space.n, env.action_space.n))
    for horizon in range(1, max_steps + 1):
        previous = result[horizon - 1].max(axis=1)
        for state, actions in env.P.items():
            for action, transitions in actions.items():
                result[horizon, state, action] = sum(
                    p * (reward + gamma * previous[nxt] * (not terminal))
                    for p, nxt, reward, terminal in transitions
                )
    return result
