import numpy as np
from scipy import spatial


def sq_obj_func(tri_data):
    x1, x2, x3 = tri_data
    return x1**2 + x2**2 + x3**2


constraint_eq = [lambda x: 1 - x[1] - x[2]]

constraint_ueq = [lambda x: 1 - x[0] * x[1], lambda x: x[0] * x[1] - 5]

def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

#region pulled from sko

def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    # generate coordinate of points randomly
    points_coordinate = np.random.rand(num_points, 2)
    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric='euclidean')

    # print('distance_matrix is: \n', distance_matrix)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    return num_points, points_coordinate, distance_matrix, cal_total_distance


def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put


def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    x1, x2 = p
    part1 = np.square(x1) - np.square(x2)
    part2 = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(part1)) - 0.5) / np.square(1 + 0.001 * part2)


def shubert(p):
    '''
    2-dimension
    -10<=x1,x2<=10
    has 760 local minimas, 18 of which are global minimas with -186.7309
    '''
    x, y = p
    part1 = [i * np.cos((i + 1) * x + i) for i in range(1, 6)]
    part2 = [i * np.cos((i + 1) * y + i) for i in range(1, 6)]
    return np.sum(part1) * np.sum(part2)


def griewank(p):
    '''
    存在多个局部最小值点，数目与问题的维度有关。
    此函数是典型的非线性多模态函数，具有广泛的搜索空间，是优化算法很难处理的复杂多模态问题。
    在(0,...,0)处取的全局最小值0
    -600<=xi<=600
    '''
    part1 = [np.square(x) / 4000 for x in p]
    part2 = [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(p)]
    return np.sum(part1) - np.prod(part2) + 1


def rastrigrin(p):
    '''
    多峰值函数，也是典型的非线性多模态函数
    -5.12<=xi<=5.12
    在范围内有10n个局部最小值，峰形高低起伏不定跳跃。很难找到全局最优
    has a global minimum at x = 0  where f(x) = 0
    '''
    return np.sum([np.square(x) - 10 * np.cos(2 * np.pi * x) + 10 for x in p])


def rosenbrock(p):
    '''
    -2.048<=xi<=2.048
    函数全局最优点在一个平滑、狭长的抛物线山谷内，使算法很难辨别搜索方向，查找最优也变得十分困难
    在(1,...,1)处可以找到极小值0
    :param p:
    :return:
    '''
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * \
               np.square(np.square(p[i]) - p[i + 1]) + np.square(p[i] - 1)
    return res

def styblinski_tang(x):
    '''
    -5 to 5 with global minimum of -39.16599*n_dim at (-2.903534,...,-2.903534)'''
    n_dim = len(x)
    sum = 0
    for i in range(n_dim - 1):
        sum += x[i] ** 4 - ( 16 * x[i] ** 2) + 5 * x[i]

    return 0.5 * sum

def sixhumpcamel(p):
    """
    带域的 2dim 的多模态全局最小化函数
    -5<=xi<=5,
    f(-0.08..., 0.712...) 的全局最小值为 -1.0...4
    """
    x, y = p
    return 4 * np.square(x) + x * y - 4 * np.square(y) - 2.1 * np.power(x, 4) + 4 * np.power(y, 4) + 1 / 3 * np.power(x,
                                                                                                                      6)


def zakharov(p):
    """
    它是一个具有范围的 n 维单峰函数
    -5<=xi<=10
    除了全局最小值之外，该函数没有局部最小值。It
    The global minimum can be found at 0, for f(0, ..., 0).
    :param p:
    :return:
    """
    temp2 = [0.5 * i * x for i, x in enumerate(p)]
    part2 = np.sum(temp2)

    temp1 = [np.square(x) for x in p]
    part1 = np.sum(temp1)
    return part1 + part2 ** 2 + part2 ** 4


def ackley(x):
    """ Ackley_N.2
    -32<=xi<=32. Convex 2dim , non-seperable function .
    The global minimum value -200 can be found at f(0,0)
    :param p:
    :return:
    """
    if not np.logical_and(x >= -32, x <= 32).all():
        raise ValueError("Input for Ackley function must be within [-32, 32].")

    d = len(x)
    j = (
        -20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum()))
        - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum())
        + 20.0
        + np.exp(1)
    )

    return j

def ani_ackley2d(x):
    """ Ackley_N.2
    -32<=xi<=32. Convex 2dim , non-seperable function .
    The global minimum value -200 can be found at f(0,0)
    :param p:
    :return:
    """

    x1, x2 = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e



def cigar(p):
    """  
    多峰全局优化函数，域为-100<=xi<=100，对于i=1...n。
    f(0,...0) 的全局最小值为 0
    """
    x = p
    return np.square(float(x[0])) + np.power(10.0, 6) * sphere(x[1:])

#endregion


#region Pulled from PySwarms

def ps_ackley(x):
    """Ackley's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-32, 32]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`


    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -32, x <= 32).all():
        raise ValueError("Input for Ackley function must be within [-32, 32].")

    d = x.shape[1]
    j = (
        -20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))
        - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
        + 20.0
        + np.exp(1)
    )

    return j

def beale(x):
    """Beale objective function.

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([3,0.5])` Its domain is bounded between :code:`[-4.5, 4.5]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Beale function only takes two-dimensional input.")
    if not np.logical_and(x >= -4.5, x <= 4.5).all():
        raise ValueError(
            "Input for Beale function must be within " "[-4.5, 4.5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
        (1.5 - x_ + x_ * y_) ** 2.0
        + (2.25 - x_ + x_ * y_ ** 2.0) ** 2.0
        + (2.625 - x_ + x_ * y_ ** 3.0) ** 2.0
    )

    return j

def booth(x):
    """Booth's objective function.

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([1,3])`. Its domain is bounded between :code:`[-10, 10]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Booth function only takes two-dimensional input.")
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError("Input for Booth function must be within [-10, 10].")

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (x_ + 2 * y_ - 7) ** 2.0 + (2 * x_ + y_ - 5) ** 2.0

    return j


def bukin6(x):
    """Bukin N. 6 Objective Function

    Only takes two dimensions and has a global minimum  of `0` at
    :code:`f([-10,1])`. Its coordinates are bounded by:
        * x[:,0] must be within [-15, -5]
        * x[:,1] must be within [-3, 3]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Bukin N. 6 function only takes two-dimensional " "input."
        )
    if not np.logical_and(x[:, 0] >= -15, x[:, 0] <= -5).all():
        raise ValueError(
            "x-coord for Bukin N. 6 function must be within " "[-15, -5]."
        )
    if not np.logical_and(x[:, 1] >= -3, x[:, 1] <= 3).all():
        raise ValueError(
            "y-coord for Bukin N. 6 function must be within " "[-3, 3]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 100 * np.sqrt(np.absolute(y_ - 0.01 * x_ ** 2.0)) + 0.01 * np.absolute(
        x_ + 10
    )

    return j


def crossintray(x):
    """Cross-in-tray objective function.

    Only takes two dimensions and has a four equal global minimums
     of `-2.06261` at :code:`f([1.34941, -1.34941])`, :code:`f([1.34941, 1.34941])`,
     :code:`f([-1.34941, 1.34941])`, and :code:`f([-1.34941, -1.34941])`.
    Its coordinates are bounded within :code:`[-10,10]`.

    Best visualized in the full domain and a range of :code:`[-2.0, -0.5]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Cross-in-tray function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError(
            "Input for cross-in-tray function must be within [-10, 10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -0.0001 * np.power(
        np.abs(
            np.sin(x_)
            * np.sin(y_)
            * np.exp(np.abs(100 - (np.sqrt(x_ ** 2 + y_ ** 2) / np.pi)))
        )
        + 1,
        0.1,
    )

    return j


def easom(x):
    """Easom objective function.

    Only takes two dimensions and has a global minimum of
    `-1` at :code:`f([pi, pi])`.
    Its coordinates are bounded within :code:`[-100,100]`.

    Best visualized in the domain of :code:`[-5, 5]` and a range of :code:`[-1, 0.2]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Easom function only takes two-dimensional input.")
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError(
            "Input for Easom function must be within [-100, 100]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = (
        -1
        * np.cos(x_)
        * np.cos(y_)
        * np.exp(-1 * ((x_ - np.pi) ** 2 + (y_ - np.pi) ** 2))
    )

    return j


def eggholder(x):
    """Eggholder objective function.

    Only takes two dimensions and has a global minimum of
    `-959.6407` at :code:`f([512, 404.3219])`.
    Its coordinates are bounded within :code:`[-512, 512]`.

    Best visualized in the full domain and a range of :code:`[-1000, 1000]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Eggholder function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -512, x <= 512).all():
        raise ValueError(
            "Input for Eggholder function must be within [-512, 512]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -(y_ + 47) * np.sin(np.sqrt(np.abs((x_ / 2) + y_ + 47))) - x_ * np.sin(
        np.sqrt(np.abs(x_ - (y_ + 47)))
    )

    return j


def goldstein(x):
    """Goldstein-Price's objective function.

    Only takes two dimensions and has a global minimum at
    :code:`f([0,-1])`. Its domain is bounded between :code:`[-2, 2]`

    Best visualized in the domain of :code:`[-1.3,1.3]` and range :code:`[-1,8000]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Goldstein function only takes two-dimensional " "input."
        )
    if not np.logical_and(x >= -2, x <= 2).all():
        raise ValueError(
            "Input for Goldstein-Price function must be within " "[-2, 2]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
        1
        + (x_ + y_ + 1) ** 2.0
        * (
            19
            - 14 * x_
            + 3 * x_ ** 2.0
            - 14 * y_
            + 6 * x_ * y_
            + 3 * y_ ** 2.0
        )
    ) * (
        30
        + (2 * x_ - 3 * y_) ** 2.0
        * (
            18
            - 32 * x_
            + 12 * x_ ** 2.0
            + 48 * y_
            - 36 * x_ * y_
            + 27 * y_ ** 2.0
        )
    )

    return j


def himmelblau(x):
    """Himmelblau's  objective function

    Only takes two dimensions and has a four equal global minimums
     of zero at :code:`f([3.0,2.0])`, :code:`f([-2.805118,3.131312])`,
     :code:`f([-3.779310,-3.283186])`, and :code:`f([3.584428,-1.848126])`.
    Its coordinates are bounded within :code:`[-5,5]`.

    Best visualized with the full domain and a range of :code:`[0,1000]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Himmelblau function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -5, x <= 5).all():
        raise ValueError(
            "Input for Himmelblau function must be within [-5,5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = (x_ ** 2 + y_ - 11) ** 2 + (x_ + y_ ** 2 - 7) ** 2

    return j


def holdertable(x):
    """Holder Table objective function

    Only takes two dimensions and has a four equal global minimums
     of `-19.2085` at :code:`f([8.05502, 9.66459])`, :code:`f([-8.05502, 9.66459])`,
     :code:`f([8.05502, -9.66459])`, and :code:`f([-8.05502, -9.66459])`.
    Its coordinates are bounded within :code:`[-10, 10]`.

    Best visualized with the full domain and a range of :code:`[-20, 0]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Holder Table function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError(
            "Input for Holder Table function must be within [-10,10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -np.abs(
        np.sin(x_)
        * np.cos(y_)
        * np.exp(np.abs(1 - np.sqrt(x_ ** 2 + y_ ** 2) / np.pi))
    )

    return j


def levi(x):
    """Levi objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([1,1])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Levi function only takes two-dimensional input.")
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError("Input for Levi function must be within [-10, 10].")

    mask = np.full(x.shape, False)
    mask[:, -1] = True
    masked_x = np.ma.array(x, mask=mask)

    w_ = 1 + (x - 1) / 4
    masked_w_ = np.ma.array(w_, mask=mask)
    d_ = x.shape[1] - 1

    j = (
        np.sin(np.pi * w_[:, 0]) ** 2.0
        + ((masked_x - 1) ** 2.0).sum(axis=1)
        * (1 + 10 * np.sin(np.pi * (masked_w_).sum(axis=1) + 1) ** 2.0)
        + (w_[:, d_] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w_[:, d_]) ** 2.0)
    )

    return j


def matyas(x):
    """Matyas objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
    """
    if not x.shape[1] == 2:
        raise IndexError("Matyas function only takes two-dimensional input.")
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError(
            "Input for Matyas function must be within " "[-10, 10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.26 * (x_ ** 2.0 + y_ ** 2.0) - 0.48 * x_ * y_

    return j


def rastrigin(x):
    """Rastrigin objective function.

    Has a global minimum at :code:`f(0,0,...,0)` with a search
    domain of :code:`[-5.12, 5.12]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -5.12, x <= 5.12).all():
        raise ValueError(
            "Input for Rastrigin function must be within " "[-5.12, 5.12]."
        )

    d = x.shape[1]
    j = 10.0 * d + (x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)

    return j


def ps_rosenbrock(x):
    """Rosenbrock objective function.

    Also known as the Rosenbrock's valley or Rosenbrock's banana
    function. Has a global minimum of :code:`np.ones(dimensions)` where
    :code:`dimensions` is :code:`x.shape[1]`. The search domain is
    :code:`[-inf, inf]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    r = np.sum(
        100 * (x.T[1:] - x.T[:-1] ** 2.0) ** 2 + (1 - x.T[:-1]) ** 2.0, axis=0
    )

    return r


def schaffer2(x):
    """Schaffer N.2 objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-100,100]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Schaffer N. 2 function only takes two-dimensional " "input."
        )
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError(
            "Input for Schaffer function must be within " "[-100, 100]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.5 + (
        (np.sin(x_ ** 2.0 - y_ ** 2.0) ** 2.0 - 0.5)
        / ((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    )

    return j


def ps_sphere(x):
    """Sphere objective function.

    Has a global minimum at :code:`0` and with a search domain of
        :code:`[-inf, inf]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = (x ** 2.0).sum(axis=1)

    return j


def threehump(x):
    """Three-hump camel objective function

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([0, 0])`. Its coordinates are bounded within
    :code:`[-5, 5]`.

    Best visualized in the full domin and a range of :code:`[0, 2000]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Three-hump camel function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -5, x <= 5).all():
        raise ValueError(
            "Input for Three-hump camel function must be within [-5, 5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = 2 * x_ ** 2 - 1.05 * (x_ ** 4) + (x_ ** 6) / 6 + x_ * y_ + y_ ** 2

    return j

#endregion Pulled from PySwarms


#Create a parameterized version of the classic Rosenbrock unconstrained optimization function
def rosenbrock_with_args(x, a, b, c=0):
    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    return f