from numba import jit
import numpy as np
import math


@jit(nopython=True, fastmath=True)
def init_w(w, n):
    """
    :purpose:
    Initialize a weight array consistent of 1s if none is given
    This is called at the start of each function containing a w param

    :params:
    w      : a weight vector, if one was given to the initial function, else None
             NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
             convert it to np.float64 (using w.astype(np.float64)) before passing it to
             any function
    n      : the desired length of the vector of 1s (often set to len(u))

    :returns:
    w      : an array of 1s with shape (n,) if w is None, else return w un-changed
    """
    if w is None:
        return np.ones(n)
    else:
        return w


@jit(nopython=True, fastmath=True)
def braycurtis(u, v, w=None):
    """
    :purpose:
    Computes the Bray-Curtis distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    braycurtis : float, the Bray-Curtis distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.braycurtis(u, v, w)
    0.3359619981199086
    """
    w = init_w(w, len(u))
    num, denom = 0, 0
    for i in range(len(u)):
        num += abs(u[i] - v[i]) * w[i]
        denom += abs(u[i] + v[i]) * w[i]
    return num / denom


@jit(nopython=True, fastmath=True)
def canberra(u, v, w=None):
    """
    :purpose:
    Computes the Canberra distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    canberra : float, the Canberra distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.canberra(u, v, w)
    1951.0399135013315
    """
    w = init_w(w, len(u))
    dist = 0
    for i in range(len(u)):
        num = abs(u[i] - v[i])
        denom = abs(u[i]) + abs(v[i])
        dist += num / denom * w[i]
    return dist


@jit(nopython=True, fastmath=True)
def chebyshev(u, v, w=None):
    """
    :purpose:
    Computes the Chebyshev distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : here, w does nothing. it is only here for consistency
             with the other functions

    :returns:
    chebyshev : float, the Chebyshev distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.chebyshev(u, v, w)
    0.9934922585052587
    """
    return max(u - v)


@jit(nopython=True, fastmath=True)
def cityblock(u, v, w=None):
    """
    :purpose:
    Computes the City Block distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    cityblock : float, the City Block distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.cityblock(u, v, w)
    1667.904767711218
    """
    w = init_w(w, len(u))
    dist = 0
    for i in range(len(u)):
        dist += abs(u[i] - v[i]) * w[i]
    return dist


@jit(nopython=True, fastmath=True)
def correlation(u, v, w=None, centered=True):
    """
    :purpose:
    Computes the correlation between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    correlation : float, the correlation between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.correlation(u, v, w)
    0.9907907248975348
    """
    w = init_w(w, len(u))
    u_centered, v_centered = u - np.mean(u), v - np.mean(v)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(len(u_centered)):
        num += u_centered[i] * v_centered[i] * w[i]
        u_norm += abs(u_centered[i]) ** 2 * w[i]
        v_norm += abs(v_centered[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return 1 - num / denom


@jit(nopython=True, fastmath=True)
def cosine(u, v, w=None):
    """
    :purpose:
    Computes the cosine similarity between two 1D arrays
    Unlike scipy's cosine distance, this returns similarity, which is 1 - distance

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    cosine  : float, the cosine similarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.cosine(u, v, w)
    0.7495065944399267
    """
    w = init_w(w, len(u))
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(len(u)):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return num / denom


@jit(nopython=True, fastmath=True)
def cosine_vector_to_matrix(u, m):
    """
    :purpose:
    Computes the cosine similarity between a 1D array and rows of a matrix

    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)

    :returns:
    cosine vector  : np.array, of shape (m,) vector containing cosine similarity between u
                     and the rows of m

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u = np.random.RandomState(seed=0).rand(10)
    >>> m = np.random.RandomState(seed=0).rand(100, 10)
    >>> fastdist.cosine_vector_to_matrix(u, m)
    (returns an array of shape (100,))
    """
    norm = 0
    for i in range(len(u)):
        norm += abs(u[i]) ** 2
    u = u / norm ** (1 / 2)
    for i in range(m.shape[0]):
        norm = 0
        for j in range(len(m[i])):
            norm += abs(m[i][j]) ** 2
        m[i] = m[i] / norm ** (1 / 2)
    return np.dot(u, m.T)


@jit(nopython=True, fastmath=True)
def cosine_matrix_to_matrix(a, b):
    """
    :purpose:
    Computes the cosine similarity between the rows of two matrices

    :params:
    a, b   : input matrices either of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1

    :returns:
    cosine matrix  : np.array, an (m, k) array of the cosine similarity
                     between the rows of a and b

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.cosine_matrix_to_matrix(a, b)
    (returns an array of shape (10, 100))
    """
    for i in range(a.shape[0]):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)
    for i in range(b.shape[0]):
        norm = 0
        for j in range(len(b[i])):
            norm += abs(b[i][j]) ** 2
        b[i] = b[i] / norm ** (1 / 2)
    return np.dot(a, b.T)


@jit(nopython=True, fastmath=True)
def cosine_pairwise_distance(a, return_matrix=False):
    """
    :purpose:
    Computes the cosine similarity between the pairwise combinations of the rows of a matrix

    :params:
    a      : input matrix of shape (n, k)
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities

    :returns:
    cosine matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                     or an (n choose 2, 1) array if return_matrix=False

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=False)
    (returns an array of shape (45, 1))

    alternatively, with return_matrix=True:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=True)
    (returns an array of shape (10, 10))
    """
    rows = np.arange(a.shape[0])
    perm = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    for i in range(a.shape[0]):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)

    if return_matrix:
        out_mat = np.zeros((len(rows), len(rows)))
        for i in range(len(rows)):
            for j in range(i):
                out_mat[i][j] = np.dot(a[i], a[j])
        return out_mat + out_mat.T
    else:
        out = np.zeros((len(perm), 1))
        for i in range(len(perm)):
            out[i] = np.dot(a[perm[i][0]], a[perm[i][1]])
        return out


@jit(nopython=True, fastmath=True)
def euclidean(u, v, w=None):
    """
    :purpose:
    Computes the Euclidean distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    euclidean : float, the Euclidean distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.euclidean(u, v, w)
    28.822558591834163
    """
    w = init_w(w, len(u))
    dist = 0
    for i in range(len(u)):
        dist += abs(u[i] - v[i]) ** 2 * w[i]
    return dist ** (1 / 2)


@jit(nopython=True, fastmath=True)
def rel_entr(x, y):
    """
    :purpose:
    Computes the relative entropy between two 1D arrays
    Used primarily for the jensenshannon function

    :params:
    x, y   : input arrays, both of shape (n,)
             to get a numerical value, x and y should be strictly non-negative;
             negative values result in infinite relative entropy

    :returns:
    rel_entr : float, the relative entropy distance of x and y
    """
    total_entr = 0
    for i in range(len(x)):
        if x[i] > 0 and y[i] > 0:
            total_entr += x[i] * math.log(x[i] / y[i])
        elif x[i] == 0 and y[i] >= 0:
            total_entr += 0
        else:
            total_entr += np.inf
    return total_entr


@jit(nopython=True, fastmath=True)
def jensenshannon(p, q, base=None):
    """
    :purpose:
    Computes the Jensen-Shannon divergence between two 1D probability arrays

    :params:
    u, v   : input probability arrays, both of shape (n,)
             note that because these are probability arrays, they are strictly non-negative
    base   : the base of the logarithm for the output

    :returns:
    jensenshannon : float, the Jensen-Shannon divergence between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).uniform(size=(10000, 2)).T
    >>> fastdist.jensenshannon(u, v, base=2)
    0.39076147897868996
    """
    p_sum, q_sum = 0, 0
    for i in range(len(p)):
        p_sum += p[i]
        q_sum += q[i]
    p, q = p / p_sum, q / q_sum
    m = (p + q) / 2
    num = rel_entr(p, m) + rel_entr(q, m)
    if base is not None:
        num /= math.log(base)
    return (num / 2) ** (1 / 2)


@jit(nopython=True, fastmath=True)
def mahalanobis(u, v, VI):
    """
    :purpose:
    Computes the Mahalanobis distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    VI     : the inverse of the covariance matrix of u and v
             note that some arrays will result in a VI containing
             very high values, leading to some imprecision

    :returns:
    mahalanobis : float, the Mahalanobis distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.array([2, 0, 0]).astype(np.float64), np.array([0, 1, 0]).astype(np.float64)
    >>> VI = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    >>> fastdist.mahalanobis(u, v, VI)
    1.7320508075688772
    """
    delta = (u - v)
    return np.dot(np.dot(delta, VI), delta) ** (1 / 2)


@jit(nopython=True, fastmath=True)
def minkowski(u, v, p, w=None):
    """
    :purpose:
    Computes the Minkowski distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    p      : the order of the norm (p=2 is the same as Euclidean)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    minkowski : float, the Minkowski distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> p = 3
    >>> fastdist.minkowski(u, v, p, w)
    7.904971256091215
    """
    w = init_w(w, len(u))
    dist = 0
    for i in range(len(u)):
        dist += abs(u[i] - v[i]) ** p * w[i]
    return dist ** (1 / p)


@jit(nopython=True, fastmath=True)
def seuclidean(u, v, V):
    """
    :purpose:
    Computes the standardized Euclidean distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    V      : array of shape (n,) containing component variances

    :returns:
    seuclidean : float, the standardized Euclidean distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, V = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.seuclidean(u, v, V)
    116.80739235578636
    """
    return euclidean(u, v, w=1 / V)


@jit(nopython=True, fastmath=True)
def sqeuclidean(u, v, w=None):
    """
    :purpose:
    Computes the squared Euclidean distance between two 1D arrays

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    sqeuclidean : float, the squared Euclidean distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.sqeuclidean(u, v, w)
    830.7398837797134
    """
    w = init_w(w, len(u))
    dist = 0
    for i in range(len(u)):
        dist += abs(u[i] - v[i]) ** 2 * w[i]
    return dist


@jit(nopython=True, fastmath=True)
def dice(u, v, w=None):
    """
    :purpose:
    Computes the Dice dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    dice : float, the Dice dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.dice(u, v, w)
    0.5008483098538385
    """
    w = init_w(w, len(u))
    num, denom = 0, 0
    for i in range(len(u)):
        num += u[i] * v[i] * w[i]
        denom += (u[i] + v[i]) * w[i]
    return 1 - 2 * num / denom


@jit(nopython=True, fastmath=True)
def hamming(u, v, w=None):
    """
    :purpose:
    Computes the Hamming distance between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    hamming : float, the Hamming distance between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.hamming(u, v, w)
    0.5061006361240681
    """
    w = init_w(w, len(u))
    num, denom = 0, 0
    for i in range(len(u)):
        if u[i] != v[i]:
            num += w[i]
        denom += w[i]
    return num / denom


@jit(nopython=True, fastmath=True)
def jaccard(u, v, w=None):
    """
    :purpose:
    Computes the Jaccard-Needham dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    jaccard : float, the Jaccard-Needham dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.jaccard(u, v, w)
    0.6674202936639468
    """
    w = init_w(w, len(u))
    num, denom = 0, 0
    for i in range(len(u)):
        if u[i] != v[i]:
            num += w[i]
            denom += w[i]
        denom += u[i] * v[i] * w[i]
    return num / denom


@jit(nopython=True, fastmath=True)
def kulsinski(u, v, w=None):
    """
    :purpose:
    Computes the Kulsinski dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    kulsinski : float, the Kulsinski dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.kulsinski(u, v, w)
    0.8325522836573094
    """
    w = init_w(w, len(u))
    num, denom = 0, 0
    for i in range(len(u)):
        num += (1 - u[i] * v[i]) * w[i]
        if u[i] != v[i]:
            num += w[i]
            denom += w[i]
        denom += w[i]
    return num / denom


@jit(nopython=True, fastmath=True)
def rogerstanimoto(u, v, w=None):
    """
    :purpose:
    Computes the Rogers-Tanimoto dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    rogerstanimoto : float, the Rogers-Tanimoto dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.rogerstanimoto(u, v, w)
    0.672067488699178
    """
    w = init_w(w, len(u))
    r, denom = 0, 0
    for i in range(len(u)):
        if u[i] != v[i]:
            r += 2 * w[i]
        else:
            denom += w[i]
    return r / (denom + r)


@jit(nopython=True, fastmath=True)
def russellrao(u, v, w=None):
    """
    :purpose:
    Computes the Ruseell-Rao dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    russelrao : float, the Russell-Rao dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.russellrao(u, v, w)
    0.7478068878987577
    """
    w = init_w(w, len(u))
    num, n = 0, 0
    for i in range(len(u)):
        num += u[i] * v[i] * w[i]
        n += w[i]
    return (n - num) / n


@jit(nopython=True, fastmath=True)
def sokalmichener(u, v, w=None):
    """
    :purpose:
    Computes the Sokal-Michener dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    sokalmichener : float, the Sokal-Michener dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.sokalmichener(u, v, w)
    0.672067488699178

    :note:
    scipy's implementation returns a different value in the above example.
    when no w is given, our implementation and scipy's are the same.
    to replicate scipy's result of 0.8046210454292805, we can replace
    r += 2 * w[i] with r += 2, but then that does not apply the weights.
    so, we use (what we think) is the correct weight implementation
    """
    w = init_w(w, len(u))
    r, s = 0, 0
    for i in range(len(u)):
        if u[i] != v[i]:
            r += 2 * w[i]
        else:
            s += w[i]
    return r / (s + r)


@jit(nopython=True, fastmath=True)
def sokalsneath(u, v, w=None):
    """
    :purpose:
    Computes the Sokal-Sneath dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    sokalsneath : float, the Sokal-Sneath dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.sokalsneath(u, v, w)
    0.8005423661929552
    """
    w = init_w(w, len(u))
    r, denom = 0, 0
    for i in range(len(u)):
        if u[i] != v[i]:
            r += 2 * w[i]
        denom += u[i] * v[i] * w[i]
    return r / (r + denom)


@jit(nopython=True, fastmath=True)
def yule(u, v, w=None):
    """
    :purpose:
    Computes the Yule dissimilarity between two boolean 1D arrays

    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    yule   : float, the Sokal-Sneath dissimilarity between u and v

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.yule(u, v, w)
    1.0244476251862624
    """
    w = init_w(w, len(u))
    ctf, cft, ctt, cff = 0, 0, 0, 0
    for i in range(len(u)):
        if u[i] != v[i] and u[i] == 1:
            ctf += w[i]
        elif u[i] != v[i] and u[i] == 0:
            cft += w[i]
        elif u[i] == v[i] == 1:
            ctt += w[i]
        elif u[i] == v[i] == 0:
            cff += w[i]
    return (2 * ctf * cft) / (ctt * cff + ctf * cft)


@jit(nopython=True, fastmath=True)
def vector_to_matrix_distance(u, m, metric, metric_name):
    """
    :purpose:
    Computes the distance between a vector and the rows of a matrix using any given metric

    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function

    distance vector  : np.array, of shape (m,) vector containing the distance between u
                       and the rows of m

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u = np.random.RandomState(seed=0).rand(10)
    >>> m = np.random.RandomState(seed=0).rand(100, 10)
    >>> fastdist.vector_to_matrix_distance(u, m, fastdist.cosine, "cosine")
    (returns an array of shape (100,))

    :note:
    the cosine similarity uses its own function, cosine_vector_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the vector and matrix heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """

    if metric_name == "cosine":
        return cosine_vector_to_matrix(u, m)

    out = np.zeros((m.shape[0]))
    for i in range(m.shape[0]):
        out[i] = metric(u, m[i])
    return out


@jit(nopython=True, fastmath=True)
def matrix_to_matrix_distance(a, b, metric, metric_name):
    """
    :purpose:
    Computes the distance between the rows of two matrices using any given metric

    :params:
    a, b   : input matrices either of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function

    :returns:
    distance matrix  : np.array, an (m, k) array of the distance
                       between the rows of a and b

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
    (returns an array of shape (10, 100))

    :note:
    the cosine similarity uses its own function, cosine_matrix_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the two matrices heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """
    if metric_name == "cosine":
        return cosine_matrix_to_matrix(a, b)

    out = np.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i][j] = metric(a[i], b[j])
    return out


@jit(nopython=True, fastmath=True)
def matrix_pairwise_distance(a, metric, metric_name, return_matrix=False):
    """
    :purpose:
    Computes the distance between the pairwise combinations of the rows of a matrix

    :params:
    a      : input matrix of shape (n, k)
    metric : the function used to calculate the distance
    metric_name   : str of the function name. this is only used for
                    the if statement because cosine similarity has its
                    own function
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities

    :returns:
    distance matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                       or an (n choose 2, 1) array if return_matrix=False

    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=False)
    (returns an array of shape (45, 1))

    alternatively, with return_matrix=True:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=True)
    (returns an array of shape (10, 10))
    """
    if metric_name == "cosine":
        return cosine_pairwise_distance(a, return_matrix)

    else:
        rows = np.arange(a.shape[0])
        perm = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
        if return_matrix:
            out_mat = np.zeros((len(rows), len(rows)))
            for i in range(len(rows)):
                for j in range(i):
                    out_mat[i][j] = metric(a[i], a[j])
            return out_mat + out_mat.T
        else:
            out = np.zeros((len(perm), 1))
            for i in range(len(perm)):
                out[i] = metric(a[perm[i][0]], a[perm[i][1]])
            return out
