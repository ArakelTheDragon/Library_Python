


                                                        # Library for working with strings, test with pytest

def repeat_a_string (str1, n):                          # Description:
                                                        # Repeat s string from var1 to var2 n times, no spaces,
                                                        # no commas and so on
                                                        # The function is tested and it works

                                                        # Local variables and initialization
                                                        # Requires the library:
                                                        # none

                                                        # Processing
    #for i in range (len (str1)):
    #    str2[i] = str1 [i]
    #    str2[i*n] = str1[i]
                                                        # Reporting
    return str1 * n                                        # Return the string


def find_string_index(str1, str2):                      # Description:
                                                        # Find str1 within str2

                                                        # Local variables and initialization
                                                        # Requires the libraries:
                                                        # import math, os,
                                                        # import pytest

                                                        # Processing
    #print()                                             # If there is no statement in the function, we
                                                        # will receive an IndentationError: expected an indented block

    string_index = str2.find(str1)                      # Find the first element's index of str1 within str2
    if  string_index != -1:
        print (string_index)

                                                        # Reporting
    return string_index                                 # return the beginning of the string(first index)
                                                        # in the second string


def create_palindrome(str1):                            # Description:
    #print()                                             # If there is no statement in the function, we
                                                        # will receive an IndentationError: expected an indented block

                                                        # Local variables and initialization
                                                        # Requires the libraries:
                                                        # import math, os
                                                        # import pytest

    rev = reversed(str1)

                                                        # Processing

    if list(str1) != list(rev):
        print ("NOT PALINDROME")
        return -1
    else:
        return 0
                                                        # Reporting
    #return 0                                            # return the palindrome for comparison with other functions