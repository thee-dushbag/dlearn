"""
A simple neural network that predicts
the chance of winning given the average
number of toes of the team.

The prediction is simply calculated by
multiplying the input data with the
weight (knob) to get the prediction.
"""

weight = 0.1


def neural_network(data: float, weight: float):
    prediction = data * weight
    return prediction


number_of_toes = 8.5, 9.5, 10, 9

for toes in number_of_toes:
    pred = neural_network(toes, weight)
    pred = int(pred * 100)
    print(f"{toes=:<3}  {pred=}%")
