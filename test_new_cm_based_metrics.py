# Author: Dr. Sascha D. Krauss
# Contact: sascha.krauss@uk-essen.de
# Start date: 2022-02-17

import timeit

import numpy as np
from sklearn import metrics

from fastdist import fastdist


def get_mpv(cm):
	"""calculate and return quality metrics based on the diagonal of the confusion matrix

	Args:
		cm: confusion matrix
	"""

	n_classes = cm.shape[0]
	cm_col = cm / cm.sum(axis=0)[np.newaxis, :]
	mpv = np.trace(cm_col) / n_classes  # Mean predictive value

	return mpv


def get_miou(cm):
	"""calculate and return quality metrics based on the diagonal of the confusion matrix

	Args:
		cm: confusion matrix
	"""

	n_classes = cm.shape[0]

	# calculate the necessary matrix for the mean Intersection over Union (IoU)
	cm_iou = np.zeros((n_classes, n_classes))
	for row in range(n_classes):
		cm_row_sum = cm[row, :].sum()
		for col in range(n_classes):
			# tp / (tp + fp + fn)
			cm_iou[row, col] = cm[row, col] / (cm_row_sum + cm[:, col].sum() - cm[row, col])

	miou = np.trace(cm_iou) / n_classes  # Mean IoU/ Jaccard

	return miou


sizes = [10000, 100000000]

for size in sizes:
	print('Vector length: ' + str(size))
	y_true = np.random.randint(2, size=size)
	y_pred = np.random.randint(2, size=size)

	print('Confusion matrix')
	start = timeit.default_timer()
	cm1 = fastdist.confusion_matrix(y_true, y_pred)
	stop = timeit.default_timer()
	duration1 = stop - start
	print('Time fastdist: ', duration1)
	cm2 = metrics.confusion_matrix(y_true, y_pred)
	stop = timeit.default_timer()
	duration2 = stop - start
	print('Time sklearn: ', duration2)
	print('Fastdist took '
		  + str(duration1 / duration2)
		  + ' times as much time as sklearn.')

	print('Overall Accuracy')
	start = timeit.default_timer()
	acc1 = fastdist.accuracy_score(y_true, y_pred, cm1)
	stop = timeit.default_timer()
	duration1 = stop - start
	print('Time: ', duration1)
	start = timeit.default_timer()
	acc2 = fastdist.accuracy_score(y_true, y_pred)
	stop = timeit.default_timer()
	duration2 = stop - start
	print('Time: ', duration2)
	print('Fastdist with precalculated confusion matrix took '
		  + str(duration1 / duration2)
		  + ' times as much time as without.')
	assert acc1 == acc2
	start = timeit.default_timer()
	acc3 = metrics.accuracy_score(y_true, y_pred)
	stop = timeit.default_timer()
	duration3 = stop - start
	print('Time: ', duration3)
	print('Fastdist took '
		  + str(duration1 / duration3)
		  + ' times as much time as sklearn.')
	assert acc2 == acc3

	print('Mean predictive value')
	start = timeit.default_timer()
	mpv1 = fastdist.mean_predictive_value(y_true, y_pred, cm1)
	stop = timeit.default_timer()
	duration1 = stop - start
	print('Time: ', duration1)
	start = timeit.default_timer()
	mpv2 = get_mpv(cm1)
	stop = timeit.default_timer()
	duration2 = stop - start
	print('Time: ', duration2)
	print('Fastdist  took '
		  + str(duration1 / duration2)
		  + ' times as much time as own function.')
	assert mpv1 == mpv2

	print('Mean Intersection over Union')
	start = timeit.default_timer()
	miou1 = fastdist.mean_iou(y_true, y_pred, cm1)
	stop = timeit.default_timer()
	duration1 = stop - start
	print('Time: ', duration1)
	start = timeit.default_timer()
	miou2 = get_miou(cm1)
	stop = timeit.default_timer()
	duration2 = stop - start
	print('Time: ', duration2)
	print('Fastdist  took '
		  + str(duration1 / duration2)
		  + ' times as much time as own function.')
	assert miou1 == miou2
