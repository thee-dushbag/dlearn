"""
With multiple inputs, the neural
network uses the knowledge stored in the
weights to transform the input to
single predictions which are then summed
together to get the overall prediction.

The most interrecting part is that, the
knowledge will know which datapoints contribue
the least to the overall predictions and they
are scaled down while the ones that matter more
are scaled up hence some factors in the data
have higher importance than others. Consider
a teams experience and perormance in the previous
games is a greater determining factor than how
popular the team is in winning the next match.
"""

from typing import NamedTuple


class Weight(NamedTuple):
    toes: float
    wins: float
    fans: float


class Data(NamedTuple):
    average_toes: float
    win_ratio: float
    fans_count: int


# The weights carry the networks knowledge
# it has accumulated so far. As we can see,
# a teams chance of winning is greatly attributed
# to how many games they have won, how many fans
# they have doesn't really matter in their
# winning the game and also, the number of toes
# they have on average does have some connections
# to their performance.
weights = Weight(0.1, 0.4, 0)


def neural_network(datum: Data, weights: Weight) -> float:
    # Each datapoint is scaled up/down
    # depending on how much impact it has
    # on the chance of the team winning
    # Note: the network does not store any data
    toes_pred = datum.average_toes * weights.toes
    win_pred = datum.win_ratio * weights.wins
    fans_pred = datum.fans_count * weights.fans
    pred = toes_pred + win_pred + fans_pred
    return pred


def neural_network2(datum: Data, weights: Weight) -> float:
    from more_itertools import dotproduct

    pred = dotproduct(datum, weights)
    return pred


data = (
    Data(8.5, 13 / 32, 20),
    Data(9.5, 13 / 16, 34),
    Data(10, 17 / 20, 56),
    Data(9.8, 3 / 4, 5),
)

for datum in data:
    prediction = neural_network(datum, weights)
    # assert prediction == neural_network2(datum, weights)
    prediction = int(prediction * 100)
    print(f"{datum=} {prediction=}%")
