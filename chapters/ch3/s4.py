"""
Each prediction needs it's own
knowledge to compute hence the
data is processed as many times
as the required number of outputs.
"""

from typing import NamedTuple
from more_itertools import dotproduct


class Predictions(NamedTuple):
    hurt: float
    win: float
    happy: float


class Data(NamedTuple):
    average_toes: float
    win_ratio: float
    fans_count: float

# Encapsulates knowledge needed to
# process some input data to some
# output prediction
class Weight(NamedTuple):
    # How toes affect some prediction
    toes: float
    # How win affect some prediction
    win: float
    # How fans affect some prediction
    fans: float


class Weights(NamedTuple):
    # Knowledge about getting hurt
    hurt: Weight
    # Knowledge about winning
    win: Weight
    # Knowledge about being happy
    # and celebrating
    happy: Weight

# What the network knows
weights = Weights(
    # High WinRatio will make the team happiest
    # Huge fan count will motivate the team
    # ToesAverage doesn't factor into happyness
    happy=Weight(0, 0.8, 0.75),
    # Winning depends greatly on thier previous wins
    # Running which affects winning is greatly
    # attributed to the state of the legs
    # A good fan base can raise the motivation
    # of a team hence factor a little bit in them winning
    win=Weight(0.45, 0.85, 0.15),
    # Playing more games will hurt the team more,
    # greater wins wins means more games they've
    # played hence higher chance of being hurt
    # Since running if directly affected by the state
    # of the legs, therefore, less toes would mean
    # less running efficiency which would hurt if they
    # kept falling from time to time and getting hurt
    # Fan count can theoretically hurt the team
    # especially if no one comes to cheer for them
    # but feelings aside, the teams capability is more
    # on team structure and coorperation
    hurt=Weight(0.1, 0.7, 0),
)


def neural_network(data: Data, weights: Weights):
    return Predictions(
        hurt=dotproduct(data, weights.hurt),
        win=dotproduct(data, weights.win),
        happy=dotproduct(data, weights.happy),
    )


data = (
    Data(8.5, 13 / 32, 20),
    Data(9.5, 13 / 16, 34),
    Data(10, 17 / 20, 56),
    Data(9.8, 3 / 4, 5),
)

for datum in data:
    pred = neural_network(datum, weights)
    print(f"{pred=} {datum=}")
