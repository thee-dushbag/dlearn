from ..ch5 import util as _u

type Network = tuple[_u.Matrix, ...]


def trannet(transformer: _u.Transformer, network: Network) -> Network:
    return tuple(_u.tranmat(transformer, matrix) for matrix in network)


def creatnet(*layers: int) -> Network:
    from itertools import pairwise as P

    def valid_layers(layers: tuple[int, ...]):
        assert len(layers) > 1, "expect atleast two layers: input and output"
        for layer in layers:
            assert isinstance(layer, int) and layer > 0, layer
            yield layer

    return tuple(_u.creatmat(o, i) for i, o in P(valid_layers(layers)))


def collapse(network: Network) -> _u.Matrix:
    from functools import reduce

    return reduce(_u.mulmat, reversed(network))
