from itertools import count
from ..ch5 import util as u, s3 as s
from .util import *
from random import Random

rand = Random(567)


def network(data: u.Vector, weights: Network) -> u.Vector:
    from functools import reduce

    return reduce(s.network, weights, data)


vals = count()

weights: Network = creatnet(2, 3, 1)
# weights = trannet(lambda _: rand.random(), weights)
weights = trannet(lambda _: next(vals), weights)
simple = collapse(weights)
print(weights)
print(simple)

datum: u.Vector = (3, 4)
pred = network(datum, weights)
pred2 = s.network(datum, simple)
print(pred, pred == pred2)
