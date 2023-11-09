import numpy as np


def sq_obj_func(tri_data):
    x1, x2, x3 = tri_data
    return x1**2 + x2**2 + x3**2


constraing_eq = [lambda x: 1 - x[1] - x[2]]

constraint_ueq = [lambda x: 1 - x[0] * x[1], lambda x: x[0] * x[1] - 5]
