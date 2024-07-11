from . import util as u
from random import Random

rand = Random()

trand: u.Transformer = lambda _: rand.random()
tsrand: u.Transformer = lambda _: 2 * rand.random() - 1


def relu(x: float) -> float:
    return x if x > 0 else 0


def relud(x: float) -> bool:
    return x > 0


def _accum_network(data: u.Vector, weights: u.Network) -> list[u.Vector]:
    layers: list[u.Vector] = [data]
    input_data: u.Matrix = (data,)
    for weight in weights:
        input_data = u.mulmat(input_data, weight.matrix)
        if weight.actfunc is not None:
            input_data = u.tranmat(weight.actfunc, input_data)
        layers.append(input_data[0])
    return layers


def network(data: u.Vector, weights: u.Network) -> u.Vector:
    input_data: u.Matrix = (data,)
    for weight in weights:
        input_data = u.mulmat(input_data, weight.matrix)
        if weight.actfunc is not None:
            input_data = u.tranmat(weight.actfunc, input_data)
    return input_data[0]


def learn2(
    weights: u.Network,
    data: u.Vector,
    goal_pred: u.Vector,
    *,
    alpha: float | None = None,
    errts: float | None = None,
) -> u.Network:
    alpha = 5e-2 if alpha is None else alpha
    errts = 5e-5 if errts is None else errts

    while True:
        preds = _accum_network(data, weights)
        delta = u.subvec(preds.pop(), goal_pred)
        deltas: list[u.Matrix] = [(delta,)]
        error = u.dotproduct(delta, delta)

        # print(f"  {error=} {delta=}")
        if error <= errts:
            break

        preds.reverse()

        for weight in reversed(weights):
            tweight = u.transpose(weight.matrix)
            delta = u.mulmat(deltas[-1], tweight)
            deltas.append(delta)
        deltas.pop()

        for weight, hdata, delta in zip(reversed(weights), preds, deltas, strict=True):
            if weight.dactfunc is not None:
                hdata = u.tranvec(weight.dactfunc, hdata)
            delta = u.scalemat(delta, alpha)
            weight_delta = u.mulmat(u.transpose((hdata,)), delta)
            weight.matrix = u.submat(weight.matrix, weight_delta)
    return weights


def train(
    weights: u.Network,
    dataset: u.Matrix,
    goal_preds: u.Matrix,
    *,
    alpha: float | None = None,
    errts: float | None = None,
    epochs: int | None = None,
) -> u.Network:
    epochs = 30 if epochs is None else epochs
    for i in range(1, epochs + 1):
        total_error: float = 0
        for data, goal_pred in zip(dataset, goal_preds, strict=True):
            weights = learn2(weights, data, goal_pred, alpha=alpha, errts=errts)
            pred = network(data, weights)
            print(f"  {pred=} {goal_pred=} {data=}")
            delta = u.subvec(pred, goal_pred)
            total_error += u.dotproduct(delta, delta)
        print(f"{i}/{epochs}: {total_error=}")
    return weights


weights = u.creatnet(2, (4, relu, relud), 1)
# weights = u.trannet(tsrand, weights)
weights = u.trannet(lambda _: 0.5, weights)
streetlights: u.Matrix = (1, 1), (0, 1), (0, 0), (1, 0)
walk_vs_stop: u.Matrix = u.transpose(((0, 1, 0, 1),))

for _ in range(2):
    print(f"{weights=}")
    for light, (gpred,) in zip(streetlights, walk_vs_stop):
        pred = network(light, weights)[0]
        print(f"{pred=:.3f} {gpred=:.3f} {light=}")
    if _ == 0:
        print()
        weights = train(
            weights, streetlights, walk_vs_stop, epochs=10, alpha=1e-4, errts=5e-5
        )
