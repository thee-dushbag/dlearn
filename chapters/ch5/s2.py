from .util import *


def network(data: float, weights: Vector) -> Vector:
    return scalevec(weights, data)


def learn(
    weights: Vector, wdeltas: Vector, alphas: Vector | float | int | None = None
) -> Vector:
    alphas = 1 if alphas is None else alphas
    if isinstance(alphas, (int, float)):
        alphas = creatvec(len(weights), alphas)
    return subvec(weights, mulvec(wdeltas, alphas))


def learn2(
    weights: Vector,
    data: float,
    goal_preds: Vector,
    *,
    errtss: Vector | float | int | None = None,
    alphas: Vector | float | int | None = None,
) -> Vector:
    errtss = 1e-10 if errtss is None else errtss
    alphas = 1e-3 if alphas is None else alphas
    if isinstance(errtss, (int, float)):
        errtss = creatvec(len(weights), errtss)
    if isinstance(alphas, (int, float)):
        alphas = creatvec(len(weights), alphas)
    while True:
        preds = network(data, weights)
        deltas = subvec(preds, goal_preds)
        errors = mulvec(deltas, deltas)
        update = tuple(map(float.__gt__, errors, errtss))
        if not any(update):
            return weights
        wdeltas = scalevec(deltas, data)
        target_wdeltas = mulvec(wdeltas, update)
        weights = learn(weights, target_wdeltas, alphas)


def main():
    wlrec_data = 0.65, 1.0, 1.0, 0.9

    goal_pred_hurt = 0.1, 0.0, 0.0, 0.1
    goal_pred_win = 1.0, 1.0, 0.0, 1.0
    goal_pred_sad = 0.1, 0.0, 0.1, 0.2

    weights: Vector = 0.3, 0.2, 0.9
    datum: float = wlrec_data[0]
    goal_preds: Vector = goal_pred_hurt[0], goal_pred_win[0], goal_pred_sad[0]

    new_weights: Vector = learn2(weights, datum, goal_preds, errtss=0, alphas=1)
    new_preds = network(datum, new_weights)

    print(f"{weights=} {new_weights=}")
    print(f"{goal_preds=} {new_preds=}")


if __name__ == "__main__":
    main()
