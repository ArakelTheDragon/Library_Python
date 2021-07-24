import math


def cmplx(real, im):
    return (real, im)


def cmplx_add(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])


def cmplx_abs(c):
    return math.sqrt(c[0] * c[0] + c[1] * c[1])


def cmplx_square(c):
    return cmplx_mult(c, c)


def cmplx_mult(c1, c2):
    a, b = c1
    c, d = c2
    return cmplx(a * c - b * d, a * d + b * c)
