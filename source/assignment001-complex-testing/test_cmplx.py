                                                          # Library includes
import math
import cmath
import random

                                                          # Library includes custom made, not from Python
import cmplx


def test_complex ():
    test_cmplx_constr ()
    test_cmplx_add ()
    test_cmplx_mult ()
    test_cmplx_abs ()
    test_cmplx_square ()


def create_random_tuple(min_value = -10, max_value = 10):
    interval_size = max_value - min_value
    return random.random() * interval_size + min_value, random.random() * interval_size + min_value


def test_cmplx_constr():
    delta_error = 1e-9
    #pairs=  [[ create_random_tuple(),  create_random_tuple()] for i in range(100)]
    tuples=  [create_random_tuple() for i in range(100)]

    for tuple in tuples:
        re1, im1 = tuple
        c1 = complex(re1, im1)
        cp1 = cmplx.cmplx(re1, im1)
        if abs(c1.real - cp1[0]) + abs(c1.imag - cp1[1]) > delta_error:
            print ("test_cmplx_constr: Error is greater")


def test_cmplx_add():
    pairs=  [[ create_random_tuple(),  create_random_tuple()] for i in range(100)]

    for pair in pairs:
        re1, im1 = pair[0]
        re2, im2 = pair[1]
        c1 = complex(re1, im1)
        cp1 = cmplx.cmplx(re1, im1)
        c2 = complex(re2, im2)
        cp2 = cmplx.cmplx(re2, im2)
        sum1 = c1 + c2
        sum1_tuple = (sum1.real, sum1.imag)
        sum2 = cmplx.cmplx_add(cp1, cp2)
        # this is important - here we compute the difference between results
        delta = abs(sum1_tuple[0] - sum2[0]) + abs(sum1_tuple[1] - sum2[1])
        some_very_small_number = 1e-9
        if delta >= some_very_small_number:
            print("test_cmplx_add: Error in addition function!", re1, im1, re2, im2, delta)


def test_cmplx_mult():
    delta_error = 1e-9
    pairs=  [[ create_random_tuple(),  create_random_tuple()] for i in range(100)]

    for pair in pairs:
                                                                            # Numbers
        re1, im1 = pair[0]
        re2, im2 = pair[1]
        c1 = complex(re1, im1)
        cp1 = cmplx.cmplx(re1, im1)
        c2 = complex(re2, im2)
        cp2 = cmplx.cmplx(re2, im2)
                                                                            # Python complex
        mult1 = c1 * c2                                                     # Result
        mult_tuple = (mult1.real, mult1.imag)
                                                                            # Complex self made
        mult2 = cmplx.cmplx_mult(cp1, cp2)
        # this is important - here we compute the difference between results
        delta = abs(mult_tuple[0] - mult2[0]) + abs(mult_tuple[1] - mult2[1])
        if delta >= delta_error:
            print("test_cmplx_mult: Error in mult function!", re1, im1, re2, im2, delta)


def test_cmplx_abs():
    delta_error = 1e-9
    tuples=  [ create_random_tuple() for i in range(100)] # Tuples not pairs

    for tuple in tuples:
                                                                            # Numbers
        #re1, im1 = tuple[0]
        c1 = complex(tuple[0], tuple[1])
        cp1 = cmplx.cmplx(tuple[0], tuple[1])

                                                                            # Python complex
        var1 = abs (c1)                                                    # Result
        #var1_tuple = (var1.real, var1.imag)
                                                                            # Complex self made
        var2 = cmplx.cmplx_abs(cp1)
        # this is important - here we compute the difference between results
        delta = abs(var1 - var2)
        if delta >= delta_error:
            print("test_cmplx_abs: Error in abs function!", var1, var2, delta)


def test_cmplx_square():
    delta_error = 1e-9
    tuples=  [ create_random_tuple() for i in range(100)]                   # Tuples not pairs

    for tuple in tuples:
                                                                            # Numbers
        #re1, im1 = tuple[0]
        #re2, im2 = tuple[1]
        c1 = complex(tuple[0], tuple[1])
        cp1 = cmplx.cmplx(tuple[0], tuple[1])

                                                                            # Python complex
        var1 = c1 * c1                                                     # Result
        var1_tuple = (var1.real, var1.imag)
                                                                            # Complex self made
        var2 = cmplx.cmplx_square(cp1)
        # this is important - here we compute the difference between results
        delta = abs(var1_tuple[0] - var2[0]) + abs(var1_tuple[1] - var2[1])
        if delta >= delta_error:
            print("test_cmplx_square: Error in square function!", var1_tuple, var2, delta)


if __name__ == "__main__":

    """
    for i in range(10):
        t1 = create_random_tuple()
        print(i, t1)
    """

    test_complex()
