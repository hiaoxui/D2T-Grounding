import numpy as np


def normalize(mat):
    """
    Normalize a matrix along the axis 1.
    :param np.ndarray mat: Matrix to normalize.
    :return np.ndarray: Normalized matrix.
    """
    if len(mat.shape) == 1:
        summed = mat.sum()
        if summed == 0:
            return np.zeros_like(mat)
        return mat / summed
    zero_lines = (mat.sum(axis=1) == 0)
    mat[zero_lines, :] += 1e-20
    mat = mat / mat.sum(axis=1).reshape(-1, 1)
    mat[zero_lines, :] = 0
    return mat


def safe_log(x):
    """
    Logarithm of tensor x.
    :param np.ndarray x: Tensor to be logged.
    :rtype: np.ndarray
    """
    non_positive = (x <= 0)
    x[non_positive] = 1e-300
    x = np.log(x)
    x[non_positive] = -float('inf')
    return x


def log_sum_exp(x):
    """
    Equivalent to \log \sum \exp (x), where the summation if done along
    the first axis.
    :param np.ndarray x: The tensor to be log_sum_exp'ed.
    :rtype: np.ndarray
    """
    base_value = x.max(axis=0)
    base_value[base_value == -float('inf')] = 1e-300
    x = x - base_value
    x = np.exp(x)
    x = x.sum(axis=0)
    if isinstance(x, np.ndarray):
        non_positive = (x <= 0)
        x[non_positive] = 1e-300
        rst = base_value + np.log(x)
        rst[non_positive] = -float('inf')
        return rst
    else:
        if x <= 0:
            return -float('inf')
        return np.log(x) + base_value


def Gaussian(mu, sigma, x):
    """
    The PDF of Gaussian distribution.
    If variance is zero, Gaussian distribution degrades to delta distribution
    :param np.ndarray mu: Mean(s).
    :param np.ndarray sigma: Variance(s).
    :param np.ndarray x: Variable(s).
    :rtype: np.ndarray
    """
    if isinstance(sigma, float):
        if sigma == 0:
            if x == mu:
                return float('inf')
            else:
                return 0
        return np.exp(-(x - mu)**2 / 2 / sigma**2) / np.sqrt(2 * np.pi) / np.abs(sigma)
    zeros = (sigma == 0)
    sigma[zeros] = 1e-10
    equals = (x == mu)
    ret = np.exp(-(x - mu)**2 / 2 / sigma**2) / np.sqrt(2 * np.pi) / np.abs(sigma)
    ret[zeros] = 0
    ret[equals * zeros] = float('inf')
    return ret


def Zeller(s):
    """
    Zeller algorithms, which could convert a date into the day of a week.
    :param str s: Format: month_day_year. E.g. 08_08_1997
    :rtype: int
    """
    year = int(s[-2:])
    month = int(s[:s.index('_')])
    day = int(s[3:5])
    if month < 3:
        month += 12
        year -= 1
    return (year+year//4+6+(26*(month+1))//10+day) % 7


def weighted_mean_std(values):
    """
    Calculate mean and variance for a series of weighted values
    :param dict values: The key of the dict is value, and the
    value is the weight
    :rtype: float, float
    """
    dim = (1,)
    single_number = False
    for value, weight in values.items():
        if not isinstance(weight, np.ndarray):
            values[value] = np.array([weight])
            single_number = True
        dim = values[value].shape

    cnt = np.zeros(shape=dim, dtype=np.float64)
    x_sum = np.zeros(shape=dim, dtype=np.float64)
    x2_sum = np.zeros(shape=dim, dtype=np.float64)

    for value, weight in values.items():
        cnt += weight
        x_sum += value * weight
        x2_sum += value * value * weight

    zero_lines = (cnt == 0)
    cnt[zero_lines] = 1e-100

    mean = x_sum / cnt
    variance = x2_sum / cnt - mean * mean

    mean[zero_lines] = float('inf')
    variance[zero_lines] = 1.0
    variance[variance < 0] = 0.0

    if single_number:
        mean = mean[0]
        variance = variance[0]

    std = np.sqrt(variance)

    return mean, std


def split_list(l, n_part):
    """

    :param list l:
    :param int n_part:
    :rtype: list[list]
    """
    ret = list()
    for start_idx in range(n_part):
        ret.append(l[start_idx::n_part])
    return ret


if __name__ == '__main__':
    _test = 1
