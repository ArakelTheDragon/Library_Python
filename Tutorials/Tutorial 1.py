# Please consider donating
# https://www.paypal.com/donate?hosted_button_id=JZXRFTC9BPWTN&source=url






# Library imports
# No libraries are needed, these includes only generic Python elements

# Global variables

# Function definitions
def Dictionary_to_list(dictionary_elements): # Description: turns a dictionary into a list if the elements of it can be converted
    res = list() # Create a list with name res and initialize it
    for sub in dictionary_elements: # For each element in the dictionary elements, append this element to the list
        res.append(dictionary_elements[sub]) # res is the list, dictionary_elements is the dictionary, sub is the element

    return res # Return the list
