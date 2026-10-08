# Detailed model state (frozenlake-state-v2)

```text
GAME: Deterministic FrozenLake

TASK:
Choose the best NEXT action from the current state below. Evaluate the future consequences of this action, assuming you can choose later actions freely. Output only its letter A, B, C, or D, with no explanation.

WORLD AND COORDINATES:
The board has 4 rows and 4 columns. Coordinates are (row, column), starting at (0, 0) in the TOP-LEFT corner. Rows increase DOWNWARD; columns increase to the RIGHT.
The full map is visible and static. There are no hidden tiles or moving obstacles. Movement is deterministic: there is NO slipping or randomness.
P = your current position on safe floor; F = safe frozen floor; H = a hole; G = the goal. P marks your position NOW, even if you started elsewhere. Any original start tile is ordinary safe floor.

MOVEMENT AND TERMINATION:
Each action attempts to move exactly ONE cell. LEFT decreases the column by 1; DOWN increases the row by 1; RIGHT increases the column by 1; UP decreases the row by 1. Diagonal moves, jumps, and wrapping around the board are impossible.
All four actions can be selected. An action beyond the board boundary leaves you in the SAME cell and still consumes one move. It does NOT end the game.
Entering F is safe and play continues. You may revisit safe cells. Entering H immediately ends the game in FAILURE: holes are not walls, so moving toward a hole makes you fall in; you cannot pass through it.
Entering G immediately ends the game in SUCCESS. After success or failure there are no further moves or rewards.

REWARD AND PLANNING OBJECTIVE:
The reward is exactly +1 on the move that enters G, and 0 on every other move, including entering H or staying at an edge. There is no additional step penalty or hole penalty.
The discount factor is gamma = 0.99. Maximize the sum of discounted future rewards: r1 + gamma*r2 + gamma^2*r3 + ... . Reaching G on the next move gives return 1; reaching it after k moves gives gamma^(k-1). Never reaching G gives return 0. For 0 < gamma < 1, a shorter safe route to G is better than a longer safe route.
This decision uses an infinite-horizon discounted objective; there is no remaining-step budget or rollout time limit in this state evaluation. If several actions have equally best return, any of them is correct.

CURRENT STATE:
Player P: row 0, column 0.
Goal G: row 3, column 3.
Map: column numbers are above; row numbers are on the left.
     0  1  2  3
 0   P  F  F  F
 1   F  H  F  H
 2   F  F  F  H
 3   H  F  F  G

ACTION CHOICES:
A: LEFT
B: DOWN
C: RIGHT
D: UP

Select exactly one next-action letter: A, B, C, or D.
```
