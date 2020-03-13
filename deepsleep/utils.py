import numpy as np
import logging
import threading
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss


class Metrics(object):
    def __init__(self, predicted_values, true_values):
        self.predicted_values = predicted_values
        self.true_values = true_values
        self.logger = logging.getLogger(__name__)

    def build_conf_matrix(self):
        """Build Confusion matrix given True values and Predicted Values.

        Returns: confusion matrix with specified labels.

        """
        pred_vals = []
        for prediction in self.predicted_values:
            pred_vals.append(np.argmax(prediction))

        print confusion_matrix(self.true_values, pred_vals, labels=[1, 2, 3, 4])

        self.logger.info("Confusion Matrix : {}".format(confusion_matrix(self.true_values, pred_vals,
                                                                         labels=[1, 2, 3, 4])))

    def get_classification_report(self):

        pred_vals = []
        for prediction in self.predicted_values:
            pred_vals.append(np.argmax(prediction))

        report = classification_report(self.true_values, pred_vals, labels=[1, 2, 3, 4])
        print report
        self.logger.info("Classification report : {}".format(report))

        return report



class ResampleDataset(object):

    def __init__(self, sampling_mode='random_over_sample'):
        self.sampling_mode = sampling_mode
        self.logger = logging.getLogger(__name__)

    def balance_classes(self, heart_signal, labels,):

        # heart_list = [item for sublist in heart_signal for item in sublist]
        heart_list = []
        labels_list = []
        for ind, sublist in enumerate(heart_signal):
            if labels[ind] < 0:
                labels[ind] = 4
            new_label_list = [int(x) for x in str(labels[ind])]*len(sublist)
            labels_list.append(new_label_list)
            for item in sublist:
                heart_list.append(item)

        flat_labels_list = [item for sublist in labels_list for item in sublist]

        self.logger.debug("Flat heart = {}, Flat labels = {}".format(len(heart_list), len(flat_labels_list)))
        assert len(heart_list) == len(flat_labels_list)

        heart_list = np.reshape(heart_list, (-1,1))

        heart_signal_res, labels_res = [], []
        if self.sampling_mode == 'under_sample':
            heart_signal_res, labels_res = self.undersample_data(X = heart_list, y = flat_labels_list)

        if self.sampling_mode == 'over_sample':
            heart_signal_res, labels_res = self.smote_oversample(X = heart_list, y = flat_labels_list)

        if self.sampling_mode == 'random_over_sample':
            heart_signal_res, labels_res = self.random_oversample(X = heart_list, y = flat_labels_list)

        return heart_signal_res, labels_res

    def undersample_data(self, X, y):
        under_sampler = NearMiss('majority', n_jobs=2)
        heart_signal_res, labels_res = under_sampler.fit_sample(X, y)

        heart_signal_res = np.reshape(heart_signal_res, (heart_signal_res.shape[0],))
        return heart_signal_res, labels_res

    def smote_oversample(self, X, y):
        sm = SMOTE(n_jobs=2)
        heart_signal_res, labels_res = sm.fit_sample(X, y)

        heart_signal_res = np.reshape(heart_signal_res, (heart_signal_res.shape[0],))
        return heart_signal_res, labels_res

    def random_oversample(self, X, y):
        ros = RandomOverSampler(random_state=42, ratio='all')
        heart_signal_res, labels_res = ros.fit_sample(X, y)

        heart_signal_res = np.reshape(heart_signal_res, (heart_signal_res.shape[0],))
        return heart_signal_res, labels_res


class ThreadSafe(object):
    """Make a generator thread safe.

    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):  # Py2
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.

    Args:
        f (generator): A generator function

    Returns:

    """
    def g(*a, **kw):
        return ThreadSafe(f(*a, **kw))
    return g