import typing as ty
from .util import *

type Data = tuple[float, ...]


def network(data: Vector, weights: Vector) -> float:
    return dotproduct(data, weights)


def learn(weights: Vector, wdeltas: Vector, alpha: float | None = None) -> Vector:
    alpha = 1 if alpha is None else alpha
    return subvec(weights, scalevec(wdeltas, alpha))


def learn2(
    weights: Vector,
    data: Vector,
    goal_pred: float,
    *,
    errts: float | None = None,
    alpha: float | None = None,
):
    errts = 1e-10 if errts is None else errts
    alpha = 1e-3 if alpha is None else alpha
    while True:
        pred = network(data, weights)
        delta = pred - goal_pred
        error = delta**2
        if error <= errts:
            return weights
        wdeltas = scalevec(data, delta)
        weights = learn(weights, wdeltas, alpha)


def main():
    toes: Data = 8.5, 9.5, 9.9, 9.0
    wlrec: Data = 0.65, 0.8, 0.8, 0.9
    nfans: Data = 1.2, 1.3, 0.5, 1.0
    data: list[Vector] = [(t, w, n) for t, w, n in zip(toes, wlrec, nfans, strict=True)]

    wl_preds: ty.Sequence[float] = 1, 1, 0, 1

    weights: Vector = 0.1, 0.2, -0.1
    datum: Vector = data[0]
    goal_pred: float = wl_preds[0]

    new_weights = learn2(weights, datum, goal_pred)
    print(f"{weights=} {new_weights=}")

    ndigits: int = 4

    pred = round(network(datum, weights), ndigits)
    new_pred = round(network(datum, new_weights), ndigits)

    print(f"{datum=} {pred=} {new_pred=}")


if __name__ == "__main__":
    main()
