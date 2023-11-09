import numpy as np
from sko.PSO import PSO
from sko.DE import DE


class HO(PSO, DE):
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
        initial_pop_tethered = 10,
        w=0.8,  # PSO
        c1=0.5,  # PSO
        c2=0.5,  # PSO
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
        verbose=False,  # PSO
        dim=None,  # PSO
        prob_mut=0.3,  # DE
        early_stop=None,  # DE
        initial_points = None,
    ):
        super().__init__(
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
        self.pso:PSO = PSO(
            func,
            n_dim=self.n_dim,
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
            n_dim=self.n_dim,
            F=F,
            size_pop=pop,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            prob_mut=prob_mut,
            constraint_eq=constraint_eq,
            constraint_ueq=constraint_ueq,
        )

        self.F = F
        self.initial_points = initial_points
        self.initial_dev = initial_dev
        self.initial_pop_tethered = initial_pop_tethered

        self.Y = np.zeros((self.pop, 1))


        self.crtbp(initial_points=initial_points, initial_deviation=initial_dev, initial_tether_pop=initial_pop_tethered)

    def crtbp(self, initial_points=None, initial_deviation = 1e2, initial_tether_pop=10):
        #create the population and set it for the first round of PSO-GA
        assert self.pop >= initial_tether_pop, "Tether population greater than overall population!"
        if initial_points is not None:
            x_free = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop-initial_tether_pop, self.n_dim))
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
            x_tethered = np.random.uniform(low=lower_tether, high=upper_tether, size=(initial_tether_pop, self.n_dim))
            self.X = np.vstack((x_free, x_tethered))

    def run(self, max_iter=None, precision=None, N=20):
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
            self.X, self.Y, self.V = self.de.X, np.reshape(self.de.Y, (self.pop, 1)), self.de.V
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
        

    fit = run

    def pso_iter():
        pass

    def de_iter():
        pass

