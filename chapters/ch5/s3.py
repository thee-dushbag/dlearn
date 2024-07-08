from .util import *

type Data = ty.Sequence[float]


def network(data: Vector, weights: Matrix) -> Vector:
    return transpose(mulmat(weights, transpose((data,))))[0]


def learn(
    weights: Matrix, wdeltas: Matrix, alphas: Matrix | float | int | None = None
) -> Matrix:
    if alphas is None:
        return submat(weights, wdeltas)
    if isinstance(alphas, (int, float)):
        return submat(weights, scalemat(wdeltas, alphas))
    return submat(weights, mulmatvec(wdeltas, alphas))


def learn2(
    weights: Matrix,
    data: Vector,
    goal_preds: Vector,
    *,
    errtss: Vector | float | int | None = None,
    alphas: Matrix | float | int | None = None,
) -> Matrix:
    errtss = 1e-10 if errtss is None else errtss
    alphas = 1e-3 if alphas is None else alphas
    if isinstance(errtss, (int, float)):
        errtss = creatvec(len(weights), errtss)
    if isinstance(alphas, (int, float)):
        alphas = creatmat(len(weights), len(weights[0]), alphas)
    while True:
        preds = network(data, weights)
        deltas = subvec(preds, goal_preds)
        errors = mulvec(deltas, deltas)
        update = tuple(map(float.__gt__, errors, errtss))
        if not any(update):
            return weights
        deltas = mulvec(deltas, update)
        wdeltas = mulmat(transpose((deltas,)), (data,))
        weights = learn(weights, wdeltas, alphas)


def main():
    weights: Matrix = (0.1, 0.1, -0.3), (0.1, 0.2, 0.0), (0.0, 1.3, 0.1)

    toes: Data = 8.5, 9.5, 9.9, 9.0
    wlrec: Data = 0.65, 0.8, 0.8, 0.9
    nfans: Data = 1.2, 1.3, 0.5, 1.0
    data: list[Vector] = [(t, w, n) for t, w, n in zip(toes, wlrec, nfans, strict=True)]

    goal_pred_hurt: Data = 0.1, 0.0, 0.0, 0.1
    goal_pred_win: Data = 1.0, 1.0, 0.0, 1.0
    goal_pred_sad: Data = 0.1, 0.0, 0.1, 0.2
    goal_preds: Vector = goal_pred_hurt[0], goal_pred_win[0], goal_pred_sad[0]

    datum: Vector = data[0]

    _new_weights = learn2(weights, datum, goal_preds)
    new_weights = tranmat(lambda c: round(c, 3), _new_weights)
    print(f"{weights=} {new_weights=}")

    new_preds = network(datum, new_weights)
    new_preds = tranvec(lambda c: round(c, 3), new_preds)
    print(f"{goal_preds=} {new_preds=}")


if __name__ == "__main__":
    main()
