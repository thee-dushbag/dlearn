import dataclasses as dt
from ..ch5.util import *


@dt.dataclass(slots=True)
class Weights:
    matrix: Matrix
    actfunc: Transformer | None = dt.field(default=None, repr=False)
    dactfunc: Transformer | None = dt.field(default=None, repr=False)


type Network = tuple[Weights, ...]
type _BaseTran[T, Transformer] = tuple[T, Transformer, Transformer]
type _LayerTran = _BaseTran[int, Transformer]


def creatnet(*layers: int | _LayerTran) -> Network:
    assert len(layers) > 1, "Must provide atleast two layers: input and output."

    def _valid_layer(layer: int | _LayerTran) -> _BaseTran[int, Transformer | None]:
        if isinstance(layer, int):
            assert layer > 0, "Invalid layer size"
            return (layer, None, None)
        assert (
            isinstance(layer[0], int) and layer[0] > 0
        ), "Expected natural (number) layer size"
        return layer

    layers_iter = map(_valid_layer, layers)
    input_size, _, _ = next(layers_iter)
    weights: list[_BaseTran[Matrix, Transformer | None]] = []

    for output_size, actfunc, dactfunc in layers_iter:
        weights.append((creatmat(input_size, output_size), actfunc, dactfunc))
        input_size = output_size

    return tuple(Weights(mat, act, dact) for mat, act, dact in weights)


def tranweight(tran: Transformer, w: Weights) -> Weights:
    return Weights(tranmat(tran, w.matrix), w.actfunc, w.dactfunc)


def trannet(transformer: Transformer, network: Network) -> Network:
    return tuple(tranweight(transformer, w) for w in network)
