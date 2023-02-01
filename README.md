# fastdist: Faster distance calculations in python using numba

fastdist is a replacement for scipy.spatial.distance that shows significant speed improvements by using numba and some optimization

Newer versions of fastdist (> 1.0.0) also add partial implementations of sklearn.metrics which also show significant speed improvements.

What's new in each version:

- 1.1.0: adds implementation of several sklearn.metrics functions, fixes an error in the Chebyshev distance calculation and adds slight speed optimizations.
- 1.1.1: large speed optimizations for confusion matrix-based metrics (see more about this in the "1.1.1 speed improvements" section), fix precision and recall scores
- 1.1.2: speed improvement and bug fix for `cosine_pairwise_distance`
- 1.1.3: bug fix for `f1_score`, which resulted from v1.1.1 speed improvements
- 1.1.4: bug fix for `float32`, speed improvements for accuracy score by allowing confusion matrix
- 1.1.5: make cosine function calculate cosine distance rather than cosine distance (as in earlier versions) for consistency with scipy, fix in-place matrix modification for cosine matrix functions

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install fastdist.

```bash
pip install fastdist
```

## Usage

For calculating the distance between 2 vectors, fastdist uses the same function calls
as scipy.spatial.distance. So, for example, to calculate the Euclidean distance between
2 vectors, run:

```python
from fastdist import fastdist
import numpy as np

u = np.random.rand(100)
v = np.random.rand(100)

fastdist.euclidean(u, v)
```

The same is true for most sklearn.metrics functions, though not all functions in sklearn.metrics are implemented in fastdist.
Notably, most of the ROC-based functions are not (yet) available in fastdist. However, the other functions are the same as sklearn.metrics.
So, for example, to create a confusion matrix from two discrete vectors, run:

```python
from fastdist import fastdist
import numpy as np

y_true = np.random.randint(10, size=10000)
y_pred = np.random.randint(10, size=10000)

fastdist.confusion_matrix(y_true, y_pred)
```

For calculating distances involving matrices, fastdist has a few different functions instead of scipy's cdist and pdist.

To calculate the distance between a vector and each row of a matrix, use `vector_to_matrix_distance`:

```python
from fastdist import fastdist
import numpy as np

u = np.random.rand(100)
m = np.random.rand(50, 100)

fastdist.vector_to_matrix_distance(u, m, fastdist.euclidean, "euclidean")
# returns an array of shape (50,)
```

To calculate the distance between the rows of 2 matrices, use `matrix_to_matrix_distance`:

```python
from fastdist import fastdist
import numpy as np

a = np.random.rand(25, 100)
b = np.random.rand(50, 100)

fastdist.matrix_to_matrix_distance(a, b, fastdist.euclidean, "euclidean")
# returns an array of shape (25, 50)
```

Finally, to calculate the pairwise distances between the rows of a matrix, use `matrix_pairwise_distance`:

```python
from fastdist import fastdist
import numpy as np

a = np.random.rand(10, 100)
fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=False)
# returns an array of shape (10 choose 2, 1)
# to return a matrix with entry (i, j) as the distance between row i and j
# set return_matrix=True, in which case this will return a (10, 10) array
```

## Speed

fastdist is significantly faster than scipy.spatial.distance in most cases.

Though almost all functions will show a speed improvement in fastdist, certain functions will have
an especially large improvement. Notably, cosine similarity is much faster, as are the vector/matrix,
matrix/matrix, and pairwise matrix calculations.

Note that numba - the primary package fastdist uses - compiles the function to machine code the first
time it is called. So, the first time you call a function will be slower than the following times, as
the first runtime includes the compile time.

Here are some examples comparing the speed of fastdist to scipy.spatial.distance:

```python
from fastdist import fastdist
import numpy as np
from scipy.spatial import distance

a, b = np.random.rand(200, 100), np.random.rand(2500, 100)
%timeit -n 100 fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
# 8.97 ms ± 11.2 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# note this high stdev is because of the first run taking longer to compile

%timeit -n 100 distance.cdist(a, b, "cosine")
# 57.9 ms ± 4.43 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

In this example, fastdist is about 7x faster than scipy.spatial.distance. This difference only gets larger
as the matrices get bigger and when we compile the fastdist function once before running it. For example:

```python
from fastdist import fastdist
import numpy as np
from scipy.spatial import distance

a, b = np.random.rand(200, 1000), np.random.rand(2500, 1000)
# i complied the matrix_to_matrix function once before this so it's already in machine code
%timeit fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
# 25.4 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit distance.cdist(a, b, "cosine")
# 689 ms ± 10.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Here, fastdist is about 27x faster than scipy.spatial.distance. Though cosine similarity is particularly
optimized, other functions are still faster with fastdist. For example:

```python
from fastdist import fastdist
import numpy as np
from scipy.spatial import distance

a = np.random.rand(200, 1000)

%timeit fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean")
# 14 ms ± 458 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit distance.pdist(a, "euclidean")
# 26.9 ms ± 1.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

fastdist's implementation of the functions in sklearn.metrics are also significantly faster. For example:

```python
from fastdist import fastdist
import numpy as np
from sklearn import metrics

y_true = np.random.randint(2, size=100000)
y_pred = np.random.randint(2, size=100000)

%timeit fastdist.accuracy_score(y_true, y_pred)
# 74 µs ± 5.81 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit metrics.accuracy_score(y_true, y_pred)
# 7.23 ms ± 157 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Here, fastdist is about 97x faster than sklearn's implementation.

#### 1.1.1 speed improvements

fastdist v1.1.1 adds significant speed improvements to confusion matrix-based metrics functions (balanced accuracy score, precision, and recall).
These speed improvements are possible by not recalculating the confusion matrix each time, as sklearn.metrics does.

In older versions of fastdist (<v1.1.1), we also recalculate the confusion matrix each time, giving us the following speed:

```python
from fastdist import fastdist
import numpy as np
from sklearn import metrics

y_true = np.random.randint(2, size=10000)
y_pred = np.random.randint(2, size=10000)

%timeit fastdist.balanced_accuracy_score(y_true, y_pred)
# 1.39 ms ± 66.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit metrics.balanced_accuracy_score(y_true, y_pred)
# 11.3 ms ± 185 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Here, fastdist is about 8x faster than sklearn.metrics.

However, now let's say that we need to compute confusion matrices and then also want to compute balanced accuracy:

```python
from fastdist import fastdist
import numpy as np
from sklearn import metrics

y_true = np.random.randint(2, size=10000)
y_pred = np.random.randint(2, size=00000)

%timeit fastdist.confusion_matrix(y_true, y_pred)
# 1.45 ms ± 55.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit metrics.confusion_matrix(y_true, y_pred)
# 11.8 ms ± 499 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

The confusion matrix computation by itself is about 8x faster with fastdist. But the larger speed improvement will come now that we don't need to
recompute the confusion matrix to calculate balanced accuracy:

```python
from fastdist import fastdist
import numpy as np
from sklearn import metrics

y_true = np.random.randint(2, size=10000)
y_pred = np.random.randint(2, size=10000)

%timeit fastdist.balanced_accuracy_score(y_true, y_pred, cm)
# 11.7 µs ± 2.12 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit metrics.balanced_accuracy_score(y_true, y_pred)
# 9.81 ms ± 1.08 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Saving the confusion matrix computation here makes fastdist's balanced accuracy score 838x faster than sklearn's.
