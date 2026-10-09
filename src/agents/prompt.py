"""Self-contained state descriptions, with no oracle-derived hints."""

import math

from src.agents.actions import DIRECTIONS, LETTERS

PROMPT_VERSION = "frozenlake-state-v2"
EPISODE_PROMPT_VERSION = "frozenlake-episode-v1"


def format_prompt(board, gamma, remaining_steps=None):
    rows = board.splitlines()
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Board must be a nonempty rectangle")
    cells = "".join(rows)
    if set(cells) - set("PFHG") or cells.count("P") != 1 or cells.count("G") != 1:
        raise ValueError("Board must contain one P, one G, and only P/F/H/G")
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("Discount must be finite and satisfy 0 <= gamma < 1")
    if remaining_steps is not None and (
        not isinstance(remaining_steps, int)
        or isinstance(remaining_steps, bool)
        or remaining_steps < 1
    ):
        raise ValueError("Remaining steps must be a positive integer")
    horizon = (
        "This decision uses an infinite-horizon discounted objective; there is "
        "no remaining-step budget or rollout time limit in this state evaluation. "
        if remaining_steps is None
        else f"You have {remaining_steps} moves remaining, INCLUDING the next move. "
        "Reach G within these moves. If the final allowed move does not reach G "
        "or enter H, the game ends in TIMEOUT with no extra reward. Reaching G "
        "on the final allowed move still counts as SUCCESS. "
        "Edge collisions and revisits also consume this remaining budget. "
    )
    height, width = len(rows), len(rows[0])
    player = divmod(cells.index("P"), width)
    goal = divmod(cells.index("G"), width)
    lines = [
        "GAME: Deterministic FrozenLake",
        "",
        "TASK:",
        "Choose the best NEXT action from the current state below. Evaluate the "
        "future consequences of this action, assuming you can choose later actions "
        "freely. Output only its letter A, B, C, or D, with no explanation.",
        "",
        "WORLD AND COORDINATES:",
        f"The board has {height} rows and {width} columns. Coordinates are "
        "(row, column), starting at (0, 0) in the TOP-LEFT corner. "
        "Rows increase DOWNWARD; columns increase to the RIGHT.",
        "The full map is visible and static. There are no hidden tiles or moving "
        "obstacles. Movement is deterministic: there is NO slipping or randomness.",
        "P = your current position on safe floor; F = safe frozen floor; "
        "H = a hole; G = the goal. P marks your position NOW, even if you started "
        "elsewhere. Any original start tile is ordinary safe floor.",
        "",
        "MOVEMENT AND TERMINATION:",
        "Each action attempts to move exactly ONE cell. LEFT decreases the column "
        "by 1; DOWN increases the row by 1; RIGHT increases the column by 1; "
        "UP decreases the row by 1. Diagonal moves, jumps, and wrapping around "
        "the board are impossible.",
        "All four actions can be selected. An action beyond the board boundary "
        "leaves you in the SAME cell and still consumes one move. It does NOT "
        "end the game.",
        "Entering F is safe and play continues. You may revisit safe cells. "
        "Entering H immediately ends the game in FAILURE: holes are not walls, "
        "so moving toward a hole makes you fall in; you cannot pass through it.",
        "Entering G immediately ends the game in SUCCESS. After success or "
        "failure there are no further moves or rewards.",
        "",
        "REWARD AND PLANNING OBJECTIVE:",
        "The reward is exactly +1 on the move that enters G, and 0 on every "
        "other move, including entering H or staying at an edge. There is no "
        "additional step penalty or hole penalty.",
        f"The discount factor is gamma = {gamma}. Maximize the sum of discounted "
        "future rewards: r1 + gamma*r2 + gamma^2*r3 + ... . Reaching G on the "
        "next move gives return 1; reaching it after k moves gives "
        "gamma^(k-1). Never reaching G gives return 0. For 0 < gamma < 1, "
        "a shorter safe route to G is better than a longer safe route.",
        horizon
        + "If several actions have equally best return, any of them is correct.",
        "",
        "CURRENT STATE:",
        f"Player P: row {player[0]}, column {player[1]}.",
        f"Goal G: row {goal[0]}, column {goal[1]}.",
        "Map: column numbers are above; row numbers are on the left.",
        "    " + " ".join(f"{c:2d}" for c in range(width)),
    ]
    lines.extend(
        f"{r:2d}  " + " ".join(f"{cell:>2}" for cell in row)
        for r, row in enumerate(rows)
    )
    lines.extend(["", "ACTION CHOICES:"])
    lines.extend(
        f"{letter}: {direction}" for letter, direction in zip(LETTERS, DIRECTIONS)
    )
    lines.extend(["", "Select exactly one next-action letter: A, B, C, or D."])
    return chr(10).join(lines) + chr(10)


def prepare_choice_tokens(tokenizer, prompt):
    if tokenizer.chat_template:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if prefix.count("<think>") > prefix.count("</think>"):
            raise ValueError("Chat template leaves an open thinking block")
    else:
        prefix = prompt
    # A newline prevents the colon from merging with the next choice token.
    prefix += "Answer:\n"
    input_ids = tokenizer.encode(prefix, add_special_tokens=False)
    choices = []
    for letter in LETTERS:
        extended = tokenizer.encode(prefix + letter, add_special_tokens=False)
        if extended[:-1] != input_ids or len(extended) != len(input_ids) + 1:
            raise ValueError(
                f"{letter} is not a single next token at the answer position"
            )
        choices.append(extended[-1])
    if len(set(choices)) != 4:
        raise ValueError("Choice token IDs must be distinct")
    return prefix, input_ids, choices
