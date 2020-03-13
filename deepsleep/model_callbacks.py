import keras.callbacks
import logging
import numpy as np
import os


class Histories(keras.callbacks.History):
    def on_train_begin(self, logs={}):
        # print self.params
        # print self.model
        self.acc = []
        self.losses = []
        self.size = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        return

    def on_batch_begin(self, batch, logs={}):
        self.size.append(logs.get('size'))
        return

    def on_batch_end(self, batch, logs={}):
        return

class ResetModel(keras.callbacks.Callback):
    def __init__(self, training_files, batch_size):

        super(ResetModel, self).__init__()
        self.training_files = training_files
        self.batch_size = batch_size
        self.ROOT = os.getcwd()
        self.DATASET_DIR = os.path.join(self.ROOT, "dataset")
        self.logger = logging.getLogger(__name__)

    def get_batch_count(self, dataset_file):

        dataset_file = os.path.join(self.DATASET_DIR, dataset_file)
        datafile = np.load(dataset_file)
        epoch_count = datafile['Total_epochs']

        batch_count = int(epoch_count / self.batch_size)
        remainder_samples = epoch_count % self.batch_size
        if remainder_samples:
            batch_count = batch_count + 1

        return batch_count

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.debug("New epoch")
        self.file_batch_count = 0
        self.file_count = 0

        self.file_batch_counter = 0

        self.file_batch_count = self.get_batch_count(self.training_files[self.file_count])
        self.file_batch_counter = self.file_batch_counter + self.file_batch_count

        self.logger.debug("Training files for epoch {} are \n {}".format(epoch, self.training_files))

    def on_batch_end(self, batch, logs=None):

        # print "Total batches being seen = {}".format(self.file_batch_counter)
        #
        # print "File and file count = {}, {}".format(self.training_files[self.file_count], self.file_count)
        # print "Callback file batches = {}".format(self.file_batch_count)
        # print "callback current batch = {}".format(batch)
        # print "batch size = {}".format(logs.get('size'))
        # print "File and file count = {}, {}".format(self.training_files[self.file_count], self.file_count)

        # Since, batch count starts from 0
        if batch == (self.file_batch_counter - 1):
            self.model.reset_states()
            self.logger.debug("Resetting states")

            if self.file_count == len(self.training_files) - 1:
                self.file_count = 0
                self.logger.debug("Reached the last training file ... Starting new epoch")
            else:
                self.file_count += 1
                self.file_batch_count = self.get_batch_count(self.training_files[self.file_count])
                self.file_batch_counter = self.file_batch_counter + self.file_batch_count