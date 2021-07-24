                                                        # Library includes
import os, math
import pytest
import string
from string import ascii_lowercase, digits
import numpy as np
import random
from random import choice

                                                        # Library includes, custom made, not from Python
import dummy                                            # Contains the library for working with strings

def test_dummy():                                       # Description:

                                                        # Runs all the functions that I implemente and compares the
                                                        # the result fro our functions to the result from the built-in
                                                        # python pytest library functions

                                                        # Local variables and initialization

                                                        # Requires the libraries:
                                                        # import os, math
                                                        # import pytest

                                                        # Processing

    test_repeat_a_string()                              # The first function to call and test afterwards with pytest
    test_find_string_index()                            # The second function to call and test afterwards with pytest
    test_create_palindrome()                            # The third function to call and test afterwards with pytest

    #print (create_random_tuple())                       # Only for testing, must be removed

                                                        # Reporting

    #return 0                                           # Error code for everything is ok


def create_random_tuple():                              # Description:

                                                        # Creates a list of strings to use with the string functions

                                                        # Local variables and initialization

                                                        # Requires the libraries:
                                                        # import os, math
                                                        # import pytest
                                                        # import numpy as np

                                                        # Processing
    chars = ascii_lowercase + digits
    lst = [''.join(choice(chars) for _ in range(2)) for _ in range(100)]
    print(lst)
    #np.random.choice(list(ascii_lowercase), (3, 5))     # (3, 5, 4) or (3, 5) is the size

                                                        # Reporting
    return lst


def test_repeat_a_string():                             # Description:

                                                        # Repeat a string from var1 to var2 n times, no spaces
                                                        # no commas and so on

                                                        # Local variables and initialization
                                                        # Requires the libraries
                                                        # none

                                                        # Processing

    str1 = create_random_tuple()
    #print(" test_repat_a_string() " + str(str1))
    for i in range(100):
        str2 = dummy.repeat_a_string (str1[i], 2)             # Self-made function
        str3 = str1[i]*2                                      # Python function

    assert str2 == str3                                 # pytest framework, test if str2 == str3

                                                        # Reporting
    return print(" test_repeat_a_string() str2 = " + str(str2) + " str3 = " + str(str3))

def test_find_string_index():                           # Description:
                                                        # Find str1 within str2

                                                        # Local variables and initialization

                                                        # Requires the libraries:
                                                        # import os, math
                                                        # import pytest
                                                        # import string
                                                        # from string import ascii_lowercase, digits
                                                        # import numpy as np
                                                        # from random import choice

                                                        # Processing

    str2 = "atoato"
    int1 = dummy.find_string_index("ato", "atoato")
    int3 = str2.find("ato")

    print(" test_find_string_index() int1 = " + str(int1) + " int3 = " + str(int3))
    assert str(int1) == str(int3)

                                                        # Reporting
    #return (" test_find_string_index() str1 = " + str(int1) + " str3 = " + str(int3))


def test_create_palindrome():                           # Description:
                                                        # Create a palindrome of a string

                                                        # Local variables and initialization
                                                        # Requires the libraries:
                                                        # import os, math
                                                        # import pytest
                                                        # import string
                                                        # from string import ascii_lowercase, digits
                                                        # import numpy as np
                                                        # import random
                                                        # from random import choice

                                                        # Processing
    int2 = dummy.create_palindrome("ada")
    str3 = "ada"
    rev = reversed("dat")

    if list(str3) != list(rev):
        int4 = -1
    else:
        int4 = 0

    print(" int4 = " + str(int4) + " int2 = " + str(int2))
    assert int4 == int2

                                                        # Reporting
    return ("test_create")

if __name__ == "__main__":
    test_dummy()