"""
You can think of this as deducing
multiple facts about something given
some property about it.

The output predictions are returned
as a group of predictions.
"""

from typing import NamedTuple


class Weight(NamedTuple):
    hurt: float
    win: float
    happy: float


class Predictions(NamedTuple):
    hurt_pred: float
    win_pred: float
    happy_pred: float


# This contains the knowledge the
# network has. There is a high chance
# the players will be hurt and happy
# if they have a high win_ratio. There
# is also a high chance they will win
# the game if they have a high win_ratio.
# Three properties deduced from the win_ratio.
weights = Weight(0.7, 0.8, 0.9)


def neural_network(win_ratio: float, weights: Weight):
    return Predictions(
        hurt_pred=win_ratio * weights.hurt,
        win_pred=win_ratio * weights.win,
        happy_pred=win_ratio * weights.happy,
    )


data = 16 / 20, 13 / 32, 8 / 13, 26 / 35

for win_ratio in data:
    predictions = neural_network(win_ratio, weights)
    print(f"{win_ratio=:.2f} {predictions=}")
