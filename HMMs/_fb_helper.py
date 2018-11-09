from utils import *


def forward(phi):
    """

    :param np.ndarray phi: [s+1, k, M(from), M(to)]
    :return np.ndarray: alpha: [s+2, k(to), M(to)]
    """
    s, k, M, _ = phi.shape
    s -= 1

    # alpha = [s+2, k(to), M(to)]
    alpha = np.zeros(shape=(s+2, k, M))
    alpha = safe_log(alpha)
    alpha[0, 0, -2] = 0.0

    offset = k
    pad = np.zeros(shape=(offset, k, M))
    pad = safe_log(pad)
    alpha = np.concatenate([pad, alpha], axis=0)

    for idx in range(s+1):
        # alpha_clip: [k(to), k(from), M(from)]
        alpha_clip = alpha[offset+idx:offset+idx-k:-1, :, :]
        # alpha_ext: [k(to), k(from), M(from), M(to)]
        alpha_ext = alpha_clip.repeat(M).reshape(k, k, M, M)
        # alpha_ext: [k(from), M(from), k(to), M(to)]
        alpha_ext = alpha_ext.transpose([1, 2, 0, 3])

        # phi_ext: [k(to), M(from), M(to), k(from)]
        phi_ext = phi[idx, :, :, :].repeat(k).reshape(k, M, M, k)
        # phi_ext: [k(from), M(from), k(to), M(to)]
        phi_ext = phi_ext.transpose([3, 1, 0, 2])

        # to_sum: [k * M(to sum up), k(to), M(to)]
        to_sum = (alpha_ext + phi_ext).reshape(k * M, k, M)

        # log_sum_exp: [k(to), M(to)]
        alpha[idx+1+offset] = log_sum_exp(to_sum)

    # alpha = [s+2, k(to), M(to)]
    alpha = alpha[offset:, :, :]

    return alpha


def backward(phi):
    """
    :param np.ndarray phi: [s+1, k, M(from), M(to)]
    :return np.ndarray: beta: [s+2, k(from), M(from)]
    """
    s, k, M, _ = phi.shape
    s -= 1

    # beta: [s+2, M(from)]
    beta = np.zeros(shape=(s+2, M), dtype=np.float64)
    beta = safe_log(beta)
    beta[s+1, M-1] = 0.0

    pad = np.zeros(shape=(k, M), dtype=np.float64)
    pad = safe_log(pad)
    beta = np.concatenate([beta, pad], axis=0)

    pad = np.zeros(shape=(k, k, M, M), dtype=np.float64)
    phi = np.concatenate([phi, pad], axis=0)

    for idx in range(s, -1, -1):
        # beta_clip: [k(to), M(to)]
        beta_clip = beta[idx+1:idx+1+k, :]
        # beta_ext: [M(from), k(to), M(to)]
        beta_ext = beta_clip\
            .repeat(M)\
            .reshape(k, M, M)\
            .transpose([2, 0, 1])

        # phi_clip: [k(to), M(from), M(to)]
        phi_clip = phi[range(idx, idx+k), range(k), :, :]
        # phi_clip: [M(from), k(to), M(to)]
        phi_clip = phi_clip.transpose([1, 0, 2])

        # to_sum: [k*M(to sum up), M(from)]
        to_sum = (beta_ext + phi_clip).reshape(M, k*M).T

        beta[idx] = log_sum_exp(to_sum)

    beta = beta[:s+2, :]
    # beta: [s+2, M(from), k(from)]
    beta = beta.repeat(k).reshape(s+2, M, k)
    # beta: [s+2, k(from), M(from)]
    beta = beta.transpose([0, 2, 1])

    return beta
