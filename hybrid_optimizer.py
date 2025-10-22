from abc import ABCMeta, abstractmethod
import numpy as np
from enum import Enum
from typing import Tuple

from sko.PSO import PSO
from sko.DE import DE
from sko.base import SkoBase

from sko.tools import func_transformer
from sko.operators import crossover, mutation, ranking, selection
from sko.operators import mutation

# region Utilities

def reflective(self, position, bounds, **kwargs):
    r"""Reflect the particle at the boundary

    This method reflects the particles that exceed the bounds at the
    respective boundary. This means that the amount that the component
    which is orthogonal to the exceeds the boundary is mirrored at the
    boundary. The reflection is repeated until the position of the particle
    is within the boundaries. The following algorithm describes the
    behaviour of this strategy:

    .. math::
        :nowrap:

        \begin{gather*}
            \text{while } x_{i, t, d} \not\in \left[lb_d,\,ub_d\right] \\
            \text{ do the following:}\\
            \\
            x_{i, t, d} =   \begin{cases}
                                2\cdot lb_d - x_{i, t, d} & \quad \text{if } x_{i,
                                t, d} < lb_d \\
                                2\cdot ub_d - x_{i, t, d} & \quad \text{if } x_{i,
                                t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                            \end{cases}
        \end{gather*}
    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(position, bounds)
    new_pos = position
    while lower_than_bound[0].size != 0 or greater_than_bound[0].size != 0:
        if lower_than_bound[0].size > 0:
            new_pos[lower_than_bound] = (
                2 * lb[lower_than_bound[0]] - new_pos[lower_than_bound]
            )
        if greater_than_bound[0].size > 0:
            new_pos[greater_than_bound] = (
                2 * ub[greater_than_bound] - new_pos[greater_than_bound]
            )
        lower_than_bound, greater_than_bound = out_of_bounds(new_pos, bounds)

    return new_pos

def periodic(self, position, bounds, **kwargs):
    r"""Sets the particles a periodic fashion

    This method resets the particles that exeed the bounds by using the
    modulo function to cut down the position. This creates a virtual,
    periodic plane which is tiled with the search space.
    The following equation describtes this strategy:

    .. math::
        :nowrap:

        \begin{gather*}
        x_{i, t, d} = \begin{cases}
                            ub_d - (lb_d - x_{i, t, d}) \mod s_d & \quad \text{if }x_{i, t, d} < lb_d \\
                            lb_d + (x_{i, t, d} - ub_d) \mod s_d & \quad \text{if }x_{i, t, d} > ub_d \\
                            x_{i, t, d} & \quad \text{otherwise}
                      \end{cases}\\
        \\
        \text{with}\\
        \\
        s_d = |ub_d - lb_d|
        \end{gather*}

    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(
        position, bounds
    )
    lower_than_bound = lower_than_bound[0]
    greater_than_bound = greater_than_bound[0]
    bound_d = np.tile(
        np.abs(np.array(ub) - np.array(lb)), (position.shape[0], 1)
    )
    bound_d = bound_d[0]
    ub = np.tile(ub, (position.shape[0], 1))[0]
    lb = np.tile(lb, (position.shape[0], 1))[0]
    new_pos = position
    if lower_than_bound.size != 0:# and lower_than_bound[1].size != 0:
        new_pos[lower_than_bound] = ub[lower_than_bound] - np.mod(
            (lb[lower_than_bound] - new_pos[lower_than_bound]),
            bound_d[lower_than_bound],
        )
    if greater_than_bound.size != 0:# and greater_than_bound[1].size != 0:
        new_pos[greater_than_bound] = lb[greater_than_bound] + np.mod(
            (new_pos[greater_than_bound] - ub[greater_than_bound]),
            bound_d[greater_than_bound],
        )
    return new_pos

def random(self, position, bounds, **kwargs):
    """Set position to random location

    This method resets particles that exeed the bounds to a random position
    inside the boundary conditions.
    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(
        position, bounds
    )
    # Set indices that are greater than bounds
    new_pos = position
    new_pos[greater_than_bound[0]] = np.array(
        [
            np.array([u - l for u, l in zip(ub, lb)])
            * np.random.random_sample((position.shape[1],))
            + lb
        ]
    )
    new_pos[lower_than_bound[0]] = np.array(
        [
            np.array([u - l for u, l in zip(ub, lb)])
            * np.random.random_sample((position.shape[1],))
            + lb
        ]
    )
    return new_pos


def out_of_bounds(position, bounds):
    """Helper method to find indices of out-of-bound positions

    This method finds the indices of the particles that are out-of-bound.
    """
    lb, ub = bounds
    greater_than_bound = np.nonzero(position > ub)
    lower_than_bound = np.nonzero(position < lb)
    return (lower_than_bound, greater_than_bound)

class Bounds_Handler(Enum):
    PERIODIC = periodic
    REFLECTIVE = reflective
    RANDOM = random

def exp_decay(start: float, end: float, iter_num: int, max_iter: int, d2=7) -> float:
    """Exponential Decay function

    Args:
        start (float): starting value for the parameter under exponential decay.
        end (float): ending value for the parameter under exponential decay.
        iter_num (int): Current iteration number.
        max_iter (int): Maximum number of iterations
        d2 (int, optional): Constant which modulates the steepness of the exponential decay function. Defaults to 7.

    Returns:
        float: value of exponential decay function for the current iteration
    """
    return (start - end) * np.exp(-d2 * iter_num / max_iter) + end

# endregion Utilities

class PSO_DE(SkoBase):
    def __init__(
        self,
        func,
        n_dim: int,
        config: dict = None,
        F: Tuple[float, float] = (0.7, 0.1),
        size_pop: int = 50,
        max_iter: int = 200,
        lb: np.ndarray = [-1000.0],
        ub: np.ndarray = [1000.0],
        w: Tuple[float, float] = (0.9, 0.4),
        c1: Tuple[float, float] = (2.5, 0.5),
        c2: Tuple[float, float] = (0.5, 2.5),
        recomb_constant: Tuple[float, float] = (0.7, 0.7),
        constraint_eq: tuple = tuple(),
        constraint_ueq: tuple = tuple(),
        n_processes: int = 1,
        taper_DE: bool = False,
        early_stop: int = None,
        initial_guesses: np.ndarray = None,
        guess_deviation: np.ndarray = [100.0],
        guess_ratio: float = 0.25,
        vectorize_func: bool = True,
        bounds_strategy: Bounds_Handler = Bounds_Handler.PERIODIC,
        mutation_strategy: str = "DE/rand/1",
        verbose: bool = False,
    ):
        """Creates a hybrid Particle Swarm (PS)-Differential Evolution (DE) Optimizer object and initializes the swarm.

        Note:
            Hyperparameter values F, recomb_constant, w, c1, and c2 can be varied strategically over the course of the
            optimization so provide both a starting and ending value for these parameters. The strategy for their variation
            is set by an argument given in the self.run method.

        Args:
            func (_type_): The heuristic function which evaluates particle fitness or, in other words, calculates the Y value for each particle.
            n_dim (int): # of dimensions of the search space/the particle positions.
            config (dict, optional): Dictionary to configure some or all optimizer hyperparameters/settings. Defaults to None.. Defaults to None.
            F (Tuple[float, float], optional): (start, end) differential weight or mutation constant for the DE step. Increasing this value increases the magnitude of the vectors which move the particles. Defaults to (0.5, 0.5).. Defaults to (0.5, 0.5).
            size_pop (int, optional): # of particles in the swarm. Defaults to 50.
            max_iter (int, optional): Maximum # of iterations. Defaults to 200.
            lb (np.ndarray, optional): lower bounds of search space. Accepts arguments of length n_dim or 1. If length is 1, the bound is used for all dimensions. Defaults to [-1000.0].
            ub (np.ndarray, optional): upper bounds of search space. Accepts arguments of length n_dim or 1. If length is 1, the bound is used for all dimensions.. Defaults to [1000.0].
            w (Tuple[float, float], optional): (start, end) inertial weight of particles for the PS step. Increasing this value encourages the particles to explore the search space. Defaults to (0.9, 0.4).
            c1 (Tuple[float, float], optional): (start, end) cognitive parameter for PS step. Represents the velocity bias towards a particle's personal best position, encourages exploration. Defaults to (2.5,0.5).
            c2 (Tuple[float, float], optional): (start, end) social parameter for PS step. Represents the velocity bias towards the swarm's global best position, encourages exploitation. Defaults to (0.5, 2.5).
            recomb_constant (Tuple[float, float], optional): (start, end) recombination constant or crossover/mutation probability for the DE step. Defaults to (0.7, 0.7). Note: lower for stability (fewer)
            constraint_eq (tuple, optional): Constraint equality. Defaults to tuple().
            constraint_ueq (tuple, optional): Constraint inequality. Defaults to tuple().
            n_processes (int, optional): # of function evaluations to run in parallel. Defaults to 1.
            taper_DE (bool, optional): If True, the optimizer will decrease the frequency of DE steps in the optimization until they reach 0 at the end of optimization, running only PS steps. Defaults to False.
            early_stop (int, optional): _description_. Defaults to None. TODO
            initial_guesses (np.ndarray, optional): Starting point for the optimization of shape (n_dim, 1). Defaults to None.
            guess_deviation (np.ndarray, optional): If initial_points given, limits how far the initial position of the tethered particles will deviate from the initial points. Accepts arguments of length n_dim or 1 of float elements. If length is 1, the bound is used for all dimensions. Defaults to 100.
            guess_ratio (float, optional): The ratio of particles which should start at positions 'tethered' to the initial_points if given. Defaults to 0.25.
            vectorize_func (bool, optional): If True, the func argument method/heuristic function will be vectorized to calculate each particle's position independently/in parallel. Defaults to True.
            bounds_strategy (Bounds_Handler, optional): the bounds handler whose strategy should handle out-of-bounds particles. Defaults to Bounds_Handler.PERIODIC.
            mutation_strategy (str, optional): The mutation strategy which should be used for the DE steps. Defaults to 'DE/rand/1'.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.func = func_transformer(func) if config.get('vectorize_func', vectorize_func) else func  # , n_processes)
        self.func_raw = func
        self.n_processes = n_processes
        self.n_dim = n_dim

        self.F_0, self.F_t = config.get("differential_weight", F)
        self.F = self.F_0
        assert (
            config.get("size_pop", size_pop) % 2 == 0
        ), "size_pop must be an even integer for GA"
        self.size_pop = config.get("size_pop", size_pop)
        self.tether_ratio = config.get("guess_ratio", guess_ratio)
        self.max_iter = config.get("max_iter", max_iter)
        self.recomb_constant_0, self.recomb_constant_t = config.get(
            "recombination_constant", recomb_constant
        )
        self.recomb_constant = self.recomb_constant_0
        self.early_stop = config.get("early_stop", early_stop)
        self.taper_DE = config.get("taper_DE", taper_DE)
        self.taper_mutation = self.F_t != self.F_0 and self.F_t != self.F
        self.bounds_handler: Bounds_Handler = config.get(
            "bounds_strategy", bounds_strategy
        )
        self.mutation_strategy = config.get("mutation_strategy", mutation_strategy)

        self.w_0, self.w_t = config.get("inertia", w)
        self.w = self.w_0
        self.cp_0, self.cp_t = config.get("cognitive", c1)
        self.cp = self.cp_0
        self.cg_0, self.cg_t = config.get(
            "social", c2
        )  # global best -- social acceleration constant
        self.cg = self.cg_0
        self.skew_social = self.cg_0 != self.cg_t and self.cg_t != self.cg

        self.Chrom = None

        self.lb = np.array(config.get("lb", lb))
        self.ub = np.array(config.get("ub", ub))
        initial_guesses = config.get("initial_guesses", initial_guesses)
        guess_deviation = config.get("guess_deviation", guess_deviation)
        guess_ratio = config.get("guess_ratio", guess_ratio)

        assert (
            self.n_dim == self.lb.size == self.ub.size
        ), "dim == len(lb) == len(ub) is not True"
        assert np.all(self.ub > self.lb), "upper-bound must be greater than lower-bound"

        self.has_constraint = bool(constraint_ueq) or bool(constraint_eq)
        self.constraint_eq = constraint_eq
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * size_pop)

        self.crt_initial(
            initial_points=np.array(initial_guesses),
            initial_deviation=guess_deviation,
            tether_ratio=guess_ratio,
        )
        v_high = self.ub - self.lb
        self.V = np.random.uniform(
            low=-v_high, high=v_high, size=(self.size_pop, self.n_dim)
        )
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = True
        self.record_value = {"X": [], "V": [], "Y": []}
        self.verbose = verbose
    #     self,
    #     func,
    #     n_dim,
    #     config = None,
    #     F=0.5,
    #     size_pop=50,
    #     max_iter=200,
    #     lb=[-1000.0],
    #     ub=[1000.0],
    #     w=0.8,
    #     c1=0.1,
    #     c2=0.1,
    #     recomb_constant=0.001,
    #     constraint_eq=tuple(),
    #     constraint_ueq=tuple(),
    #     n_processes=0,
    #     taper_DE=False,
    #     taper_mutation=False,
    #     skew_social=True,
    #     early_stop=None,
    #     initial_guesses=None,
    #     guess_deviation=100,
    #     guess_ratio=0.25,
    #     vectorize_func=True,
    #     bounds_strategy:Bounds_Handler=Bounds_Handler.PERIODIC,
    #     mutation_strategy = 'DE/rand/1'

   

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)

    def crt_initial(
        self, initial_points=None, initial_deviation=1e2, tether_ratio=0.25
    ):
        # create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio
        num_tethered = np.floor(self.size_pop * tether_ratio)
        if initial_points is not None:
            x_free = np.random.uniform(
                low=self.lb,
                high=self.ub,
                size=(int(self.size_pop - num_tethered), self.n_dim),
            )
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
            x_tethered = np.random.uniform(
                low=lower_tether,
                high=upper_tether,
                size=(int(num_tethered), self.n_dim),
            )
            self.X = np.vstack((x_free, x_tethered))
        else:
            self.X = np.random.uniform(
                low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
            )

    # def pso_add(self, c, x1, x2):
    #     x1, x2 = x1.tolist(), x2.tolist()
    #     ind1, ind2 = np.random.randint(0, self.n_dim - 1, 2)
    #     if ind1 >= ind2:
    #         ind1, ind2 = ind2, ind1 + 1

    #     part1 = x2[ind1:ind2]
    #     part2 = [i for i in x1 if i not in part1]  # this is very slow

    #     return np.array(part1 + part2)

    def update_pso_V(self):
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        self.V = (
            self.w * self.V
            + self.cp * r1 * (self.pbest_x - self.X)
            + self.cg * r2 * (self.gbest_x - self.X)
        )
        if (self.V == 0).all():
            print("uh oh")

    def update_X(self):
        self.X = self.X + self.V
        for particle, coord in enumerate(self.X):
            if (coord < self.lb).any() or (coord > self.ub).any():
                self.X[particle] = self.bounds_handler(self, coord, (self.lb, self.ub))



    # def tsp_update_X(self):
    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         x = self.pso_add(self.cp, x, self.pbest_x[i])
    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         x = self.pso_add(self.cg, x, self.gbest_x)
    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         new_x_strategy = np.random.randint(3)
    #         if new_x_strategy == 0:
    #             x = mutation.swap(x)
    #         elif new_x_strategy == 1:
    #             x = mutation.reverse(x)
    #         elif new_x_strategy == 2:
    #             x = mutation.transpose(x)

    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    def cal_y(self):

        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        """
        personal best
        :return:
        """
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        global best
        :return:
        """
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X)
        self.record_value["Y"].append(self.Y)

    def old_run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            # self.update_V()
            self.recorder()
            self.update_X()
            # self.cal_y()
            # self.update_pbest()
            # self.update_gbest()

            if self.verbose:
                print(
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def de_iter(self):
        self.mutation()
        self.recorder()
        self.crossover()
        self.selection()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def pso_iter(self):
        self.update_pso_V()
        self.recorder()
        old_x = self.X.copy()
        self.update_X()
        if (old_x == self.X).all():
            print("this is unholy")
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def mutation(self):
        """
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        from differential evolution
        """
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        while (r1 == r2).all() or (r2 == r3).all() or (r1 == r3).all():
            random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))
            r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        match self.mutation_strategy:
            case 'DE/best/1':
                # DE/best/k strategy makes more sense here  (k=1 or 2)
                self.V = self.gbest_x + self.F * (X[r2, :] - X[r3, :])
            case 'DE/rand/1':
                self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])
            case 'DE/rand/2':
                self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # 这里F用固定值，为了防止早熟，可以换成自适应值

        # DE/either-or could also work

        # DE/cur-to-best/1 !!

        # DE/cur-to-pbest

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
        )
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        """
        if rand < prob_crossover, use V, else use X
        """
        mask = np.random.rand(self.size_pop, self.n_dim) <= self.recomb_constant
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        """
        greedy selection
        """
        X = self.X.copy()
        f_X = (
            self.x2y().copy()
        )  # Uses x2y, which incorporates the constraint equations as a large penalty
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X

    def x2y(self):
        self.cal_y()
        if self.has_constraint:
            penalty_eq = 1e5 * np.array(
                [
                    np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq]))])
                    for x in self.X
                ]
            )
            penalty_eq = np.reshape(penalty_eq, (-1, 1))
            penalty_ueq = 1e5 * np.array(
                [
                    np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq]))
                    for x in self.X
                ]
            )
            penalty_ueq = np.reshape(penalty_ueq, (-1, 1))
            self.Y_penalized = self.Y + penalty_eq + penalty_ueq
            return self.Y_penalized
        else:
            return self.Y

    def run(self, max_iter=None, precision=None, N=20, strategy:str = "exp_decay") -> Tuple[np.ndarray, float]:
        """Run the hybrid optimizer until maximum iterations or precision reached.

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to None and uses self.max_iter else uses max_iter and replaces sef.max_iter.
            precision (float, optional): If precision is None, it will run the number of max_iter steps. If precision is a float, the loop will stop if continuous N difference between pbest less than precision. Defaults to None.
            N (int, optional): # of stagnant iterations before precision is considered reached. Defaults to 20.
            strategy (str, optional): Strategy by which to vary optimization hyperparamaters. Defaults to 'exp_decay'.

        Returns:
            Tuple[np.ndarray, float]: (best position, best heuristic value) results of optimization
        """
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso_iter()

            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y) #TODO: MF can later adapt this to have multiple convergence criterion supported (like the one in Q2MM HO)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.taper_DE and self.mutation_strategy != '':
                if (
                    iter_num <= np.floor(0.25 * self.max_iter)
                    or (
                        iter_num <= np.floor(0.75 * self.max_iter)
                        and iter_num % 10 == 0
                    )
                    or (iter_num % 100 == 0)
                ):
                    self.de_iter()
            else:
                self.de_iter()

            if self.verbose:
                (
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )
            self.gbest_y_hist.append(self.gbest_y)

            if self.taper_mutation:
                if strategy == "exp_decay":
                    self.F = exp_decay(self.F_0, self.F_t, iter_num, self.max_iter)
            if self.skew_social:
                if strategy == "exp_decay":
                    self.cp = exp_decay(self.cp_0, self.cp_t, iter_num, self.max_iter)
                    self.w = exp_decay(self.w_0, self.w_t, iter_num, self.max_iter)
                    self.cg = (self.cg_0 + self.cp_0) - self.cp

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass