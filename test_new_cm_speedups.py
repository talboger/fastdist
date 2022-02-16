import timeit

import numpy as np
from sklearn import metrics

from fastdist import fastdist

y_true = np.random.randint(2, size=10000)
y_pred = np.random.randint(2, size=10000)

cm1 = fastdist.confusion_matrix(y_true, y_pred)
cm2 = metrics.confusion_matrix(y_true, y_pred)

start = timeit.default_timer()
acc1 = fastdist.accuracy_score(y_true, y_pred, cm1)
stop = timeit.default_timer()
print('Time: ', stop - start)
start = timeit.default_timer()
acc2 = metrics.accuracy_score(y_true, y_pred)
stop = timeit.default_timer()
print('Time: ', stop - start)
