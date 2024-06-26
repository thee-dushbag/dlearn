import typing as ty

type Vector = ty.Sequence[float]
type Matrix = ty.Sequence[Vector]
type Transformer = ty.Callable[[float], float]


def dotproduct(vec1: Vector, vec2: Vector, /) -> float:
    return sum(mulvec(vec1, vec2))


def scalevec(vector: Vector, scalar: float, /) -> Vector:
    return tuple(scalar * coord for coord in vector)


def scalemat(matrix: Matrix, scalar: float, /) -> Matrix:
    return tuple(scalevec(row, scalar) for row in matrix)


def addvec(vec1: Vector, vec2: Vector, /) -> Vector:
    return tuple(v1 + v2 for v1, v2 in zip(vec1, vec2, strict=True))


def addmat(mat1: Matrix, mat2: Matrix, /) -> Matrix:
    return tuple(addvec(row1, row2) for row1, row2 in zip(mat1, mat2, strict=True))


def subvec(vec1: Vector, vec2: Vector, /) -> Vector:
    return addvec(vec1, scalevec(vec2, -1))


def submat(mat1: Matrix, mat2: Matrix, /) -> Matrix:
    return addmat(mat1, scalemat(mat2, -1))


def mulvec(vec1: Vector, vec2: Vector, /) -> Vector:
    return tuple(v1 * v2 for v1, v2 in zip(vec1, vec2, strict=True))


def transpose(matrix: Matrix, /) -> Matrix:
    return tuple(col for col in zip(*matrix, strict=True))


def mulmat(mat1: Matrix, mat2: Matrix, /) -> Matrix:
    assert len(mat1[0]) == len(mat2), "columns(mat1) != rows(mat2)"
    return tuple(tuple(dotproduct(row, col) for col in transpose(mat2)) for row in mat1)


def mulmatvec(mat1: Matrix, mat2: Matrix, /) -> Matrix:
    return tuple(mulvec(row1, row2) for row1, row2 in zip(mat1, mat2, strict=True))


def recivec(vector: Vector, /) -> Vector:
    return tuple(1 / coord for coord in vector)


def recimat(matrix: Matrix, /) -> Matrix:
    return tuple(map(recivec, matrix))


def divvec(vec1: Vector, vec2: Vector, /) -> Vector:
    return mulvec(vec1, recivec(vec2))


def creatvec(size: int, init: float | None = None) -> Vector:
    init = 0 if init is None else init
    return tuple(init for _ in range(size))


def creatmat(rows: int, cols: int, init: float | None = None) -> Matrix:
    return tuple(creatvec(cols, init) for _ in range(rows))


def createmat2(rows: int, cols: int, init: Vector) -> Matrix:
    assert len(init) == rows * cols, "size mismatch"
    it = iter(init)
    return tuple(tuple(next(it) for _ in range(cols)) for _ in range(rows))


def flatmat(matrix: Matrix, /) -> Vector:
    return tuple(coord for row in matrix for coord in row)


def tranvec(transformer: Transformer, vector: Vector, /) -> Vector:
    return tuple(map(transformer, vector))


def tranmat(transformer: Transformer, matrix: Matrix, /) -> Matrix:
    return tuple(tranvec(transformer, row) for row in matrix)
