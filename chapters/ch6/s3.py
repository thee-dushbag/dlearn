from pprint import pprint
from ..ch5 import util as u
from random import SystemRandom as Random
from .util_old import Network, creatnet, trannet

rand = Random()

trand: u.Transformer = lambda _: rand.random()


def relu(x: float) -> float:
    return max(0, x)


def relud(x: float) -> float:
    return x > 0


streetlights: u.Matrix = (1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 1, 1)
streetlights: u.Matrix = (1, 0), (0, 1), (0, 0), (1, 1)
walk_vs_stop: u.Vector = 1, 1, 0, 0

alpha: float = 5e-1
error_threshold: float = 1e-30
weights: Network = trannet(trand, creatnet(2, 6, 1))
weights = trannet(lambda w: 2 * w - 1, weights)

weights_0_1: u.Matrix = weights[0]
weights_1_2: u.Matrix = weights[1]

counter: int = 0
while True:
    prev_total: float = 0.0
    total_error: float = 0.0
    counter += 1
    for data, goal_pred in zip(streetlights, walk_vs_stop):
        layer_0 = u.transpose((data,))
        layer_1 = u.mulmat(weights_0_1, layer_0)
        layer_1 = u.tranmat(relu, layer_1)
        layer_2 = u.mulmat(weights_1_2, layer_1)

        layer_2_delta: float = layer_2[0][0] - goal_pred
        total_error += layer_2_delta**2

        layer_2_delta *= alpha

        _w12_delta = u.scalemat(layer_1, layer_2_delta)

        _w01_delta = u.scalemat(u.transpose(weights_1_2), layer_2_delta)
        _w01_delta = u.mulmatvec(_w01_delta, u.tranmat(relud, layer_1))
        _w01_delta = u.mulmat(_w01_delta, u.transpose(layer_0))

        weights_1_2 = u.submat(weights_1_2, u.transpose(_w12_delta))
        weights_0_1 = u.submat(weights_0_1, _w01_delta)

    if counter == 10:
        print(total_error.hex())
        counter = 0
    
    if total_error <= error_threshold:
        break

    if total_error == prev_total:
        print("Stuck, randomize.")
        weights_0_1 = u.tranmat(trand, weights_0_1)
        weights_1_2 = u.tranmat(trand, weights_1_2)

    prev_total = total_error

pprint(weights_0_1)
pprint(weights_1_2)

for data, goal_pred in zip(streetlights, walk_vs_stop):
    layer_0 = u.transpose((data,))
    layer_1 = u.mulmat(weights_0_1, layer_0)
    layer_1 = u.tranmat(relu, layer_1)
    layer_2 = u.mulmat(weights_1_2, layer_1)
    pred = layer_2[0][0]

    print(f"{pred=:.4f} {goal_pred=:.4f} {data=}")


def _seereluatwork():
    layer_0: u.Matrix = u.transpose((streetlights[0],))
    print(layer_0)
    layer_1: u.Matrix = u.mulmat(weights[0], layer_0)
    print(layer_1)
    layer_1 = u.tranmat(relu, layer_1)
    print(layer_1)
    layer_2: u.Matrix = u.mulmat(weights[1], layer_1)
    print(layer_2)
