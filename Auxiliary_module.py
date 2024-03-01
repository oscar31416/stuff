##############################################################
# This module contains the functions which have been created #
# for an easier calculation of the characteristics of the    #
# magma chambers using the Mogi model.                       #
##############################################################

import math
import numpy
#import numba

#>>>>>
# Volume of a sphere.
#>>>>>
#@numba.njit(fastmath = True)
def volume(r):
    return ((4 / 3) * math.pi * math.pow(r, 3))


#>>>>>
# Radius change.
#>>>>>
#@numba.njit(fastmath = True)
def delta_radius(arr):
    V_end = volume(arr[1]) + arr[0]
    r_end = math.pow((V_end * (3 / 4) / math.pi), (1 / 3))
    return r_end - arr[1]


#>>>>>
# This function decides whether the chambers are too close or not.
#>>>>>
#@numba.njit(nogil = True, fastmath = True)
def check_function(shallow, deep):
    """
    This function takes the coordinates of the chambers and helps decide
    whether it's safe (or not) to calculate the results of a possible Mogi
    model. If the result is False, some models will NOT de calculated.
    """
    value = math.sqrt(math.pow(deep[3] - shallow[3], 2) + math.pow(deep[4] - shallow[4], 2) + \
                      math.pow(shallow[2], 2))
    if value < (shallow[1] + deep[1]):
        return False
    elif shallow[2] + 1000 <= shallow[1]:
        return False
    elif shallow[2] + 1000 <= deep[1]:
        return False
    elif deep[1] <= shallow[1]:
        return False
    try:
        """
        The second chamber is the one whose volume decreases. However,
        its final radius must NOT be negative as it makes no sense.
        """
        drd = delta_radius(deep)
        if drd < 0.0:
            drd = (-1.0) * drd
        if drd > deep[1]:
            return False
    except:
        return False
    """
    If no problems have been identified, the model is safe to calculate.
    """
    return True


#>>>>>
# Mogi model function.
#>>>>>
#@numba.njit(nogil = True, fastmath = True)
def mogi(data_dict, position, nu = 0.25):
    """
    This function returns a 3-element list which describes how an element which
    was initially located at 'position' moves as a result of the
    presence of a spherical magma chamber located beneath the surface.

    position_list - Point's old coordinates.
    nu - Poisson's coefficient.

    data_dict - Array that contains all the data related to the chamber.

        0 - Change in volume.
        1 - Radius.
        2 - Depth.
        3 - X coordinate.
        4 - Y coordinate.

        delta_P - Chamber's pressure increase.
        mu - Magma's shear modulus. Common for both chambers.
    """
    rel_depth = data_dict[2] + position[2]
    out = numpy.zeros(3)
    out[2] = (1.0 - nu) * math.pow(data_dict[1], 3) * \
        (delta_radius(data_dict) / data_dict[1]) * 4 * rel_depth
    out[0] = position[0] - data_dict[3]
    out[1] = position[1] - data_dict[4]
    r = math.sqrt(math.pow(out[0], 2.0) + math.pow(out[1], 2.0))

    out[2] /= math.pow((rel_depth * rel_depth) + (r * r), 1.5)
    aaa = out[2] / rel_depth
    out[0] = aaa * out[0]
    out[1] = aaa * out[1]
    return out

#>>>>>
# Calculation of theoretical end points.
#>>>>>
#@numba.njit(fastmath = True)
def deformation(initial_array, dict_1, dict_2):
    """
    This function uses the previously defined 'mogi' function to estimate the
    deformation that is caused by the two magma chambers.
    The effects of the chambers are calculated independently, and added later.
    """
    end_array = numpy.zeros(initial_array.shape)
    a = numpy.zeros(3)
    for i in range(len(initial_array)):
        for j in range(len(a)):
           a[j] = initial_array[i][j]
        end_array[i] = mogi(dict_1, a) + mogi(dict_2, a)
    return end_array


#>>>>>
# Calculation of the mean squared error.
#>>>>>
#@numba.njit(nogil = True, fastmath = True)
def error(observed, theoretical):
    """
    This function takes the displacement vectors for a set of points
    (their sizes must be the same) and does the following for each
    theoretical -- observed couple:
    > The vector that represents the difference between those vectors is
      calculated.
    > Its modulus is elevated to the power of two.
    > The function returns the square root of the sum of all the squared moduli.
    """
    total = 0.0
    for i in range(len(theoretical)):
        for j in range(len(theoretical[0, :])):
            total += math.pow((theoretical[i, j] - observed[i, j]), 2)
    return math.sqrt(total)
