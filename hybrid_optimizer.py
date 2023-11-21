from abc import ABCMeta, abstractmethod
import numpy as np
from sko.PSO import PSO
from sko.DE import DE
from sko.base import SkoBase

from sko.tools import func_transformer
from sko.operators import crossover, mutation, ranking, selection
from sko.operators import mutation


class PSO_GA(SkoBase):
    def __init__(self, func, n_dim, F=0.5, size_pop=50, max_iter=200, lb=-1000.0, ub=1000.0, w=0.8, c1=0.1, c2=0.1, prob_mut=0.001, constraint_eq=tuple(), constraint_ueq=tuple(), n_processes=0, early_stop=None, initial_guesses=None, guess_deviation=100, guess_ratio=0.25):
        self.func = func_transformer(func) #, n_processes)
        self.func_raw = func
        self.n_dim = n_dim
        self.F = F
        assert size_pop % 2 == 0, 'size_pop must be an even integer for GA'
        self.size_pop = size_pop
        self.tether_ratio = guess_ratio
        self.max_iter = max_iter
        self.prob_mut = prob_mut
        self.early_stop = early_stop

        self.w = w
        self.cp = c1 # personal best -- cognitive
        self.cg = c2 # global best -- social

        self.Chrom = None

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq) or bool(constraint_eq)
        self.constraint_eq = constraint_eq
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * size_pop)

        self.crt_initial(initial_points=initial_guesses, initial_deviation=guess_deviation, tether_ratio=guess_ratio)
        v_high = ub - lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.size_pop, self.n_dim))
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.verbose = False

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)
    
    def crt_initial(self, initial_points=None, initial_deviation = 1e2, tether_ratio=0.25):
        #create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio
        num_tethered = np.floor(self.size_pop * tether_ratio)
        if initial_points is not None:
            x_free = np.random.uniform(low=self.lb, high=self.ub, size=(int(self.size_pop-num_tethered), self.n_dim))
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
            x_tethered = np.random.uniform(low=lower_tether, high=upper_tether, size=(int(num_tethered), self.n_dim))
            self.X = np.vstack((x_free, x_tethered))
        else:
            self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))

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
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)
        
    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub) #TODO change boundary handler

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
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)

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
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

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
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def mutation(self):
        '''
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        #assert r1 != r2 and r2 != r3

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        if rand < prob_crossover, use V, else use X
        '''
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        '''
        greedy selection
        '''
        X = self.X.copy()
        f_X = self.x2y().copy() # Uses x2y, which incorporates the constraint equations as a large penalty
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X
    
    def x2y(self):
        self.cal_y()
        if self.has_constraint:
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y_penalized = self.Y + 1e5 * penalty_eq + 1e5 * penalty_ueq
            return self.Y_penalized
        else:
            return None
    
    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso_iter()

            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                ('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))
            self.gbest_y_hist.append(self.gbest_y)

            self.de_iter()

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
    
    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass

    




class HO(SkoBase, metaclass=ABCMeta):
    def __init__(
        self,
        func,
        n_dim=None,
        F=0.5,  # DE
        pop=40,  # pop in PSO, size_pop in GA
        max_iter=150,
        lb=-1e5,  # -1 in DE
        ub=1e5,  # 1 in DE
        initial_dev = 1e2,
        tether_ratio = 0.25,
        w=0.8,  # PSO
        c1=0.5,  # PSO
        c2=0.5,  # PSO
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
        verbose=False,  # PSO
        dim= None, #PSO
        prob_mut=0.3,  # DE
        early_stop=None,  # DE
        initial_points = None,
    ):
        super().__init__()
        self.pso:PSO = PSO(
            func,
            n_dim=n_dim,
            pop=pop,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            w=w,
            c1=c1,
            c2=c2,
            constraint_eq=constraint_eq,
            constraint_ueq=constraint_ueq,
            verbose=verbose,
            dim=dim
        )
        self.de:DE = DE(
            func,
            n_dim=n_dim,
            F=F,
            size_pop=pop,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            prob_mut=prob_mut,
            constraint_eq=constraint_eq,
            constraint_ueq=constraint_ueq,
        )

        self.n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.size_pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.F = F
        self.initial_points = initial_points
        self.initial_dev = initial_dev
        self.tether_ratio = tether_ratio

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        #self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.crtbp(initial_points=initial_points, initial_deviation=initial_dev, tether_ratio=tether_ratio)
        
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.size_pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

        self.prob_mut = prob_mut  # probability of mutation
        self.early_stop = early_stop

        # constraint:
        #self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        #self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        #self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []


    # @abstractmethod
    # def chrom2x(self, Chrom):
    #     pass

    # @abstractmethod
    # def ranking(self):
    #     pass
        

    def crtbp(self, initial_points=None, initial_deviation = 1e2, tether_ratio=0.25):
        #create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio
        num_tethered = np.floor(self.size_pop * tether_ratio)
        if initial_points is not None:
            x_free = np.random.uniform(low=self.lb, high=self.ub, size=(int(self.size_pop-num_tethered), self.n_dim))
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
            x_tethered = np.random.uniform(low=lower_tether, high=upper_tether, size=(int(num_tethered), self.n_dim))
            self.X = np.vstack((x_free, x_tethered))
        else:
            self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))

    def mutation(self):
        '''
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        #assert r1 != r2 and r2 != r3

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        if rand < prob_crossover, use V, else use X
        '''
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        '''
        greedy selection
        '''
        X = self.X.copy()
        f_X = self.x2y().copy()
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X

    
    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y
    

    def old_run(self, max_iter=None, precision=None, N=20):
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso.X, self.pso.Y, self.pso.V = self.X, self.Y, self.V
            self.pso.recorder()
            self.pso.update_pbest()
            self.pso.update_gbest()
            self.pso.run(max_iter=1, precision=precision, N=N)
            self.X, self.Y, self.V = self.pso.X, self.pso.Y, self.pso.V
            self.recorder()
            self.pso.need_update
            self.update_pbest()
            self.update_gbest()
            de_y = self.Y.flatten()
            self.de.X, self.de.Y, self.de.V = self.X, self.Y.flatten(), self.V
            self.de.generation_best_X.append(self.gbest_x)
            self.de.generation_best_Y.append(self.gbest_y)
            self.de.all_history_Y.append(self.de.Y)
            self.de.run(max_iter=1)
            self.X, self.Y, self.V = self.de.X, np.reshape(self.de.Y, (self.size_pop, 1)), self.de.V
            self.recorder()
            self.need_update
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c+1
                    if c> N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
        

    def update_pso_V(self):
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub) #TODO change boundary handler

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y
    
    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True
    
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso_iter()

            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                ('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))
            self.gbest_y_hist.append(self.gbest_y)

            self.de_iter()

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
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()


    fit = run

