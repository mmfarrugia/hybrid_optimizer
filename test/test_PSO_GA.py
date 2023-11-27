import os
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

src_dir = os.path.abspath("/home/mmfarrugia/repos/optimization/hybrid_optimizer")
sys.path.append(src_dir)

from hybrid_optimizer import PSO_GA
import example_funcs
#from tests.test_resources import *


ip_example = np.array([1., 1., 1.])

def test_init():
    ho = PSO_GA(example_funcs.sq_obj_func, n_dim=3, constraint_eq=example_funcs.constraint_eq, constraint_ueq=example_funcs.constraint_ueq)
    ho.X
    print()


def test_initialIn():
    ho = PSO_GA(example_funcs.sq_obj_func, n_dim=3, constraint_eq=example_funcs.constraint_eq, constraint_ueq=example_funcs.constraint_ueq, initial_guesses=ip_example)
    ho.record_mode = True
    ho.X
    num_within = 0
    num_outside = 0
    for coord in ho.X:
        i = 0
        while i < ho.n_dim and coord[i] < ip_example[i] + 1e2 and coord[i] > ip_example[i] - 1e2:
            if i == ho.n_dim - 1:
                num_within += 1
            i += 1
    assert num_within >= np.floor(ho.tether_ratio * ho.size_pop)
    
        
def test_basic_run():
    ho = PSO_GA(example_funcs.sq_obj_func, n_dim=3, constraint_eq=example_funcs.constraint_eq, constraint_ueq=example_funcs.constraint_ueq)
    ho.record_mode = True
    ho.run()

    print('best_x is ', ho.gbest_x, 'best_y is', ho.gbest_y)


    plt.plot(ho.gbest_y_hist)
    plt.show()

    print()
    

