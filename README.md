# fastdist: Faster distance calculations in python using numba

fastdist is a replacement for scipy.spatial.distance that shows significant speed improvements by using numba and some optimization

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

For calculating distances involving matrices, fastdist has a few different functions.

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

fastdist is significantly faster than scipy.spatial.distance in most cases. It also returns the exact
same values as scipy.spatial.distance (with the exception of some floating point differences), meaning
the calculations are set up correctly.

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
