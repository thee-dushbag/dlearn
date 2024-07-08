from ..ch5 import util as u, s1 as s


def train(
    weights: u.Vector,
    dataset: u.Matrix,
    goal_preds: u.Vector,
    *,
    errts: float | None = None,
    terrts: float | None = None,
    alpha: float | None = None,
) -> u.Vector:
    terrts = 0 if terrts is None else terrts

    prev_total = 0
    while True:
        total_error = 0
        for data, goal_pred in zip(dataset, goal_preds, strict=True):
            weights = s.learn2(weights, data, goal_pred, errts=errts, alpha=alpha)
            total_error += (s.network(data, weights) - goal_pred) ** 2
        if total_error <= terrts or prev_total == total_error:
            break
        prev_total = total_error
    return weights


def main():
    # Correlated Data
    streetlights_data: u.Matrix = (
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
    )

    walk_vs_stop_preds: u.Vector = 0, 1, 0, 1, 1, 0

    # Decorrelated Data: The network is too simple to learn this.
    # streetlights_data = (1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 1, 1)
    # walk_vs_stop_preds = 1, 1, 0, 0

    # initial weights is random
    weights: u.Vector = 10, 30, 40

    # Low Key Training
    # for _ in range(100):
    #     for data, goal_pred in zip(streetlights_data, walk_vs_stop_preds):
    #         weights = s.learn2(weights, data, goal_pred, errts=1e-23, alpha=1e-2)
    #         error = (s.network(data, weights) - goal_pred) ** 2
    #         print(f"{weights=} {(data, goal_pred)=} {error=}")
    #     print()

    # for data, goal_pred in zip(streetlights_data, walk_vs_stop_preds):
    #     pred = s.network(data, weights)
    #     error = (pred - goal_pred) ** 2
    #     print(f"{pred=} {goal_pred=} {data=}  {error=}")

    # Train the network on the data.
    weights = train(
        weights, streetlights_data, walk_vs_stop_preds, errts=1e-28, alpha=5e-1
    )

    for datum, goal_pred in zip(streetlights_data, walk_vs_stop_preds):
        pred = s.network(datum, weights)
        print(f"{pred=:.3f} {goal_pred=:.3f}")


if __name__ == "__main__":
    main()
