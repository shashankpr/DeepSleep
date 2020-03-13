import os
import logging
import numpy as np

from keras.preprocessing import sequence

from utils import threadsafe_generator
from preprocess import PreProcessData

seed = 42
np.random.seed(seed)

class HeartSequenceGenerator(object):

    def __init__(self, seq_len, batch_size, n_classes, is_pretrain=False, is_stateful_train=False):
        self.ROOT = os.getcwd()
        self.DATASET_DIR = os.path.join(self.ROOT, "dataset")
        self.PRETRAIN_DIR = os.path.join(self.ROOT, "pretrain_dataset")
        self.logger = logging.getLogger(__name__)

        self.is_pretrain = is_pretrain
        self.is_stateful_train = is_stateful_train
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_classes = n_classes

        self.preprocess = PreProcessData(seq_len=self.seq_len, batch_size=self.batch_size, n_classes=self.n_classes)

    def total_batch_count(self, dataset_files):

        n_batches = 0
        for npz_files in dataset_files:
            if self.is_pretrain:
                npz_files = os.path.join(self.PRETRAIN_DIR, npz_files)

                datafile = np.load(npz_files)
                sampling_rate = datafile["Sampling_rate"]
                total_samples = datafile["Total_samples"]
                # max_len = 30 * int(np.mean(sampling_rate))
                # epoch_count = int(total_samples / max_len)
                epoch_count = total_samples

            else:
                npz_files = os.path.join(self.DATASET_DIR, npz_files)
                datafile = np.load(npz_files)
                epoch_count = datafile['Total_epochs']

            current_batch_count = int(epoch_count / self.batch_size)
            remainder_samples = epoch_count % self.batch_size
            if remainder_samples:
                current_batch_count = current_batch_count + 1

            n_batches = n_batches + current_batch_count
            # self.logger.debug("Current batch count = {}".format(current_batch_count))

        self.logger.debug("Total batches to check = {}".format(n_batches))
        return n_batches


    @threadsafe_generator
    def generate_sequences(self, subject_files):

        while True:
            for files in subject_files:
                if self.is_pretrain:
                    npz_file = os.path.join(self.PRETRAIN_DIR, files)
                else:
                    npz_file = os.path.join(self.DATASET_DIR, files)

                heart_signal_seq, labels_categorical = self._preprocessor(npz_file)

                for heart_signal_batch, labels_batch in self._get_data_in_batches(heart_signal_seq, labels_categorical):
                    yield (heart_signal_batch, labels_batch)

    def validation_sequence(self, subject_files):

        val_heart = np.zeros(shape=(1, self.seq_len, 1))
        val_labels = np.zeros(shape=(1, self.n_classes))

        np.random.shuffle(subject_files)
        for files in subject_files:
            # Load .npz dataset

            if self.is_pretrain:
                npz_file = os.path.join(self.PRETRAIN_DIR, files)
                self.logger.info("Extracting Pretrain Validation {}".format(npz_file))
            else:
                npz_file = os.path.join(self.DATASET_DIR, files)
                self.logger.info("Extracting Original Validation data {}".format(npz_file))

            heart_signal_seq, labels_categorical = self._preprocessor(npz_file)

            if self.is_stateful_train:
                # make the batch sizes same across each batch
                self.logger.debug("Heart shape = {}".format(heart_signal_seq.shape))
                remaining_samples = heart_signal_seq.shape[0] % self.batch_size
                if remaining_samples:
                    heart_signal_seq, labels_categorical = self.preprocess.pad_validation_batches(heart_signal_seq,
                                                                                                  labels_categorical,
                                                                                                  remaining_samples)

            val_heart = np.vstack((val_heart, heart_signal_seq))
            val_labels = np.vstack((val_labels, labels_categorical))

        val_heart = val_heart[1:]
        val_labels = val_labels[1:]

        return val_heart, val_labels

    def _preprocessor(self, npz_file):

        heart_signal, labels, sampling_rate = self.preprocess.load_data(npz_file)
        # self.logger.debug("Heart signal shape = {}".format(heart_signal.shape))

        # Standardize the heart signal for every 30 seconds epoch
        # self.logger.info("Standardizing signals...")
        # heart_signal = self.preprocess.standardize_data(heart_signal)
        # self.logger.debug("Heart shape after standardization = {}".format(heart_signal.shape))
        #
        # # Create sequences
        # self.logger.info("Creating and Padding sequences...")
        # heart_signal_seq, labels_seq = self.preprocess.create_sequences(heart_signal, labels, sampling_rate=sampling_rate)
        # self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))
        #
        #
        # # Pad sequences to get uniform sequence length
        # heart_signal_seq = sequence.pad_sequences(heart_signal_seq, maxlen=self.seq_len, dtype='float32',
        #                                           padding='post',
        #                                           truncating='post')
        # self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))

        # Convert labels to categorical format
        self.logger.info("Converting labels to categorical...")
        labels_categorical = self.preprocess.convert_to_categorical(labels)

        # # Add extra dimension to suit the requirements for LSTM & CNN
        # if self.is_pretrain:
        #     heart_signal_seq = np.expand_dims(heart_signal_seq, 2)


        self.logger.debug("Shape = {}, {}".format(heart_signal.shape, labels_categorical.shape))

        return heart_signal, labels_categorical

    def _get_data_in_batches(self, heart_signal_seq, labels_categorical):

        if self.is_pretrain:
            indexes = self.preprocess.get_exploration_order(heart_signal=heart_signal_seq, shuffle=True)
        else:
            indexes = self.preprocess.get_exploration_order(heart_signal=heart_signal_seq)

        max_batches = self.preprocess.get_current_batch_count(index=indexes)
        for i in range(max_batches):
            if i == max_batches - 1:

                # If last batch having lesser samples than batch_size
                batch_indexes = indexes[i * self.batch_size:]
                heart_signal_batch = [heart_signal_seq[k] for k in batch_indexes]
                labels_batch = [labels_categorical[k] for k in batch_indexes]

                heart_signal_batch = np.asarray(heart_signal_batch)
                labels_batch = np.asarray(labels_batch)

                if self.is_stateful_train:
                    # make the batch sizes same across each batch for rendering statefulness
                    # Pad last batch
                    if heart_signal_batch.shape[0] != self.batch_size:
                        heart_signal_batch, labels_batch = self.preprocess.pad_batches(heart_signal=heart_signal_batch, labels=labels_batch)
            else:
                batch_indexes = indexes[i * self.batch_size: (i + 1) * self.batch_size]
                heart_signal_batch = [heart_signal_seq[k] for k in batch_indexes]
                labels_batch = [labels_categorical[k] for k in batch_indexes]

                heart_signal_batch = np.asarray(heart_signal_batch)
                labels_batch = np.asarray(labels_batch)

            yield heart_signal_batch, labels_batch


class HeartSequenceLoader(object):
    def __init__(self, seq_len, batch_size, n_classes, is_pretrain=False):

        self.ROOT = os.getcwd()
        self.DATASET_DIR = os.path.join(self.ROOT, "dataset")
        self.PRETRAIN_DIR = os.path.join(self.ROOT, "pretrain_dataset")
        self.logger = logging.getLogger(__name__)

        self.is_pretrain = is_pretrain
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_classes = n_classes

        self.preprocess = PreProcessData(seq_len=self.seq_len, batch_size=self.batch_size, n_classes=self.n_classes)

    def _preprocessor(self, npz_file):

        heart_signal, labels, sampling_rate = self.preprocess.load_data(npz_file)

        # Create sequences
        self.logger.info("Creating and Padding sequences...")
        heart_signal_seq, labels_seq = self.preprocess.create_sequences(heart_signal, labels,
                                                                        sampling_rate=sampling_rate)

        # Standardize the heart signal for every 30 seconds epoch
        heart_signal_seq = self.preprocess.standardize_seq_data(heart_signal_seq)

        # Pad sequences to get uniform sequence length
        heart_signal_seq = sequence.pad_sequences(heart_signal_seq, maxlen=self.seq_len, dtype='float32',
                                                  padding='post',
                                                  truncating='post')

        # Convert labels to categorical format
        labels_categorical = self.preprocess.convert_to_categorical(labels_seq)

        # Add extra dimension to suit the requirements for LSTM & CNN
        heart_signal_seq = np.expand_dims(heart_signal_seq, 2)
        self.logger.debug("Shape = {}, {}".format(heart_signal_seq.shape, labels_categorical.shape))

        return heart_signal_seq, labels_categorical


class InferenceHeartSequenceLoader(object):
    def __init__(self, seq_len, batch_size, n_classes, is_pretrain=False):
        self.ROOT = os.getcwd()
        self.TEST_DATASET_DIR = os.path.join(self.ROOT, "deepsleep/test_dataset")
        self.PRETRAIN_DIR = os.path.join(self.ROOT, "pretrain_dataset")
        self.logger = logging.getLogger(__name__)

        self.is_pretrain = is_pretrain
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_classes = n_classes

        self.preprocess = PreProcessData(seq_len=self.seq_len, batch_size=self.batch_size, n_classes=self.n_classes)

    def _preprocessor(self, npz_file):
        heart_signal, labels, sampling_rate = self.preprocess.load_data(npz_file)

        # Create sequences
        self.logger.info("Creating and Padding sequences...")
        heart_signal_seq, labels_seq = self.preprocess.create_sequences(heart_signal, labels,
                                                                        sampling_rate=sampling_rate)

        # Standardize the heart signal for every 30 seconds epoch
        heart_signal_seq = self.preprocess.standardize_seq_data(heart_signal_seq)

        # Pad sequences to get uniform sequence length
        heart_signal_seq = sequence.pad_sequences(heart_signal_seq, maxlen=self.seq_len, dtype='float32',
                                                  padding='post',
                                                  truncating='post')

        # Convert labels to categorical format
        # labels_categorical = self.preprocess.convert_to_categorical(labels_seq)

        # Add extra dimension to suit the requirements for LSTM & CNN
        # heart_signal_seq = np.expand_dims(heart_signal_seq, 2)
        self.logger.debug("Shape = {}, {}".format(heart_signal_seq.shape, labels_seq.shape))

        return heart_signal_seq, labels_seq

    def get_data(self, subject_file):
        """

        Args:
            subject_file:

        Returns:

        """

        npz_file = os.path.join(self.TEST_DATASET_DIR, subject_file)
        heart_signal_seq, labels = self._preprocessor(npz_file)

        return heart_signal_seq, labels

