import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical


class PreProcessData(object):
    def __init__(self, seq_len, batch_size, n_classes):
        self.logger = logging.getLogger(__name__)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_classes = n_classes

    def load_data(self, npz_file):
        datafile = np.load(npz_file)
        heart_signal = datafile['Filtered_Heart']
        labels = datafile['Labels']
        sampling_rate = datafile["Sampling_rate"]

        return heart_signal, labels, sampling_rate

    def convert_to_categorical(self, labels):

        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        categorical_labels = to_categorical(labels_encoded, num_classes=self.n_classes)
        return categorical_labels

    def standardize_seq_data(self, heart_signal):
        """
        Standardizer to scale sequential data.
        Args:
            heart_signal: Sequential signal in the form of list of lists or array of lists

        Returns:
            heart_signal_norm_seq: Normalized heart signal in the form of array of lists

        """

        intra_scaler = StandardScaler(with_mean=True, with_std=True, copy=False)

        len_list = []
        for sub_signals in heart_signal:
            len_list.append(len(sub_signals))
            # sub_signals = np.reshape(sub_signals, (-1, 1))
            # intra_scaler.fit_transform(sub_signals)

        heart_flat_list = [item for sublist in heart_signal for item in sublist]
        heart_flat_list = np.asarray(heart_flat_list)
        heart_flat_list = np.reshape(heart_flat_list, (-1, 1))
        intra_scaler.fit_transform(heart_flat_list)
        np.squeeze(heart_flat_list, axis=1)

        # Change back to sequence
        heart_signal_norm_seq = []
        count = 0
        for val in len_list:
            sublists = heart_flat_list[count: (count + val)]
            heart_signal_norm_seq.append(sublists)
            count = count + val

        heart_signal_norm_seq = np.asarray(heart_signal_norm_seq)

        return heart_signal_norm_seq

    def standardize_data(self, heart_signal):


        # Scaler for complete subject standardization
        intra_scaler = StandardScaler(with_mean=True, with_std=True, copy=False)

        if type(heart_signal[0]) is np.float64 or type(heart_signal[0]) is np.float32:
            #heart_signal = [item for sublist in heart_signal for item in sublist]
            heart_signal = np.reshape(heart_signal, (-1, 1))
            intra_scaler.fit_transform(heart_signal)
            return np.squeeze(heart_signal, axis=1)
        else:
            heart_signal_norm = self.standardize_seq_data(heart_signal)
            return heart_signal_norm

    def create_sequences(self, X, y, sampling_rate=None, sliding_window=0):
        """
        Convert list of samples to list of lists (of sequences)
        Args:
            X: input of single dimensional, non-iterable list of scalars
            seq_len: number of values to be considered in one sequence

        Return:
            X_seq: array of lists (list of sequences, each of len of `seq_len`)
        """

        X_seq = []
        y_seq = []

        if sampling_rate is not None:
            if type(sampling_rate) is list or type(sampling_rate) is np.ndarray:
                sampling_rate = int(np.mean(sampling_rate))
            seq_len = 30 * sampling_rate  # Take window sizes in multiples of 30 second samples
        else:
            seq_len = self.seq_len

        # self.logger.info("Sampling rate = {}, seq_len = {}".format(sampling_rate, seq_len))

        if type(X[0]) is np.float64 or type(X[0]) is np.float32:
            # When X is 1D list or array (pretrain data)

            # Check if there would be any left out samples due to unequal window size
            remaining_samples = int(len(X) % seq_len)
            total_sequences = int(len(X) / seq_len)

            if remaining_samples:
                total_sequences = total_sequences + 1

            # self.logger.debug("Total sequences to be created = {}, Samples left = {}".format(total_sequences, remaining_samples))
            for count in range(total_sequences):
                if count == total_sequences - 1:
                    X_seq.append(X[count * seq_len:])
                    y_seq.append(y[count * seq_len])
                else:
                    X_seq.append(X[count * seq_len:(count + 1) * seq_len])
                    y_seq.append(y[count * seq_len])

            X_seq = np.asarray(X_seq)
            y_seq = np.asarray(y_seq)

            return X_seq, y_seq

        else:
            return X, y

    def create_subsequences(self, X, y):

        subseq_heart = []
        subseq_labels = []
        for index in range(len(X) - self.seq_len):
            subseq_heart.append(X[index: index + self.seq_len])
            subseq_labels.append(y[index])

        return subseq_heart, subseq_labels

    def pad_batches(self, heart_signal, labels):

        # self.logger.debug("Original batch size = {}".format(heart_signal.shape[0]))
        samples_to_pad = self.batch_size - heart_signal.shape[0]
        self.logger.debug("Samples to pad = {}".format(samples_to_pad))
        if samples_to_pad <= -1:
            samples_to_pad = self.batch_size - heart_signal[-1].shape[0]

        padded_heart_signal = np.zeros((samples_to_pad, heart_signal.shape[1]))
        padded_heart_signal = np.expand_dims(padded_heart_signal, axis=2)

        padded_labels = np.ones(samples_to_pad) * 3
        padded_labels_categorical = to_categorical(padded_labels, 4)

        heart_signal_batch = np.concatenate((heart_signal, padded_heart_signal))
        labels_batch = np.concatenate((labels, padded_labels_categorical))

        # self.logger.debug("New batch size = {}".format(heart_signal_batch.shape[0]))
        assert heart_signal_batch.shape[0] == self.batch_size
        assert labels_batch.shape[0] == self.batch_size

        return heart_signal_batch, labels_batch

    def pad_validation_batches(self, heart_signal, labels, remaining_samples):

        # self.logger.debug("Original batch size = {}".format(heart_signal.shape[0]))
        samples_to_pad = self.batch_size - remaining_samples
        self.logger.debug("Samples to pad = {}".format(samples_to_pad))

        padded_heart_signal = np.zeros((samples_to_pad, heart_signal.shape[1]))
        padded_heart_signal = np.expand_dims(padded_heart_signal, axis=2)

        padded_labels = np.ones(samples_to_pad) * 3
        padded_labels_categorical = to_categorical(padded_labels, 4)

        heart_signal_batch = np.concatenate((heart_signal, padded_heart_signal))
        labels_batch = np.concatenate((labels, padded_labels_categorical))

        # self.logger.debug("New batch size = {}".format(heart_signal_batch.shape[0]))
        assert heart_signal_batch.shape[0] % self.batch_size == 0
        assert labels_batch.shape[0] % self.batch_size == 0

        return heart_signal_batch, labels_batch

    def get_exploration_order(self, heart_signal, shuffle=False):

        indexes = np.arange(len(heart_signal))
        if shuffle:
            np.random.shuffle(indexes)

        return indexes

    def get_current_batch_count(self, index):

        max_batches = int(len(index) / self.batch_size)
        remainder_samples = len(index) % self.batch_size
        if remainder_samples:
            max_batches = max_batches + 1

        return max_batches