import os, glob, time
import logging
import numpy as np
from collections import Counter
from sklearn.utils import class_weight

class DataLoader(object):

    def __init__(self, split_criterion = 0.3, is_pretrain=False):
        self.logger = logging.getLogger(__name__)
        self.ROOT = os.getcwd()
        self.DATASET_DIR = os.path.join(self.ROOT, "dataset")
        self.PRETRAIN_DIR = os.path.join(self.ROOT, "pretrain_dataset")
        self.split_criterion = split_criterion
        self.is_pretrain = is_pretrain

    def __get_dataset_paths(self, randomize_files=False):
        file_names = []

        for paths in glob.glob(self.DATASET_DIR + "/*.npz"):
            file_names.append(paths.split('/')[-1])

        if randomize_files:
            np.random.shuffle(file_names)
        return file_names

    def split_dataset(self, val_count = 0, randomize_files=False):

        dataset_names = self.__get_dataset_paths(randomize_files=randomize_files)
        no_of_files = len(dataset_names)

        if val_count > 0:
            training_files = dataset_names[val_count:]
            validation_files = dataset_names[0:val_count]

        else:
            validation_file_count = int(no_of_files * self.split_criterion)
            training_file_count = no_of_files - validation_file_count

            training_files = dataset_names[:training_file_count]
            validation_files = dataset_names[training_file_count: no_of_files]

        self.logger.info("Number of Training files = {}, Number of Validation files = {}".format(len(training_files), len(validation_files)))
        return training_files, validation_files


    def get_training_report(self, training_data):

        count = Counter()
        self.logger.info("Training Data Report")
        for npz_files in training_data:
            if self.is_pretrain:
                npz_files = os.path.join(self.PRETRAIN_DIR, npz_files)
            else:
                npz_files = os.path.join(self.DATASET_DIR, npz_files)

            datafile = np.load(npz_files)
            labels = datafile['Labels']
            count.update(labels)
            self.logger.info("Class distribution = {} for file = {}".format(Counter(labels), npz_files))

        total_labels = np.sum(count.values())
        self.logger.info("Total Class distribution = {}".format(count))
        self.logger.info("Total labels = {}".format(total_labels))
        self.logger.info("Light = {}, Deep = {}, REM = {}, Wake = {}".format(count[2], count[1], count[3], count[4]))
        self.logger.info(
            "%Light = {}, %Deep = {}, %REM = {}, %Wake = {}".format((count[2]/float(total_labels))*100,
                                                                          (count[1]/float(total_labels))*100,
                                                                          (count[3]/float(total_labels))*100,
                                                                          (count[4]/float(total_labels))*100
                                                                          ))

    def get_validation_report(self, validation_X, validation_y):
        count_val = Counter(validation_y)
        self.logger.info("Validation Data Report")

        # labels = validation_y
        # count_val.update(labels)
        total_labels = np.sum(count_val.values())
        self.logger.info("Total Class distribution = {}".format(count_val))
        self.logger.info("Total labels = {}".format(total_labels))
        self.logger.info(
            "%Light = {}, %Deep = {}, %REM = {}, %Wake = {}".format((count_val[2] / float(total_labels)) * 100,
                                                                    (count_val[1] / float(total_labels)) * 100,
                                                                    (count_val[3] / float(total_labels)) * 100,
                                                                    (count_val[4] / float(total_labels)) * 100
                                                                    ))

        # self.logger.info("Baseline accuracy = {}".format(np.max((count_val[2] / float(total_labels)) * 100,
        #                                                             (count_val[1] / float(total_labels)) * 100,
        #                                                             (count_val[3] / float(total_labels)) * 100,
        #                                                             (count_val[4] / float(total_labels)) * 100
        #                                                             )))

    def get_dataset_report(self, dataset):

        count = Counter()
        self.logger.info("Dataset Report")
        for npz_files in dataset:
            if self.is_pretrain:
                npz_files = os.path.join(self.PRETRAIN_DIR, npz_files)
            else:
                npz_files = os.path.join(self.DATASET_DIR, npz_files)

            datafile = np.load(npz_files)
            labels = datafile['Labels']
            count.update(labels)
            self.logger.info("Class distribution = {} for file = {}".format(Counter(labels), npz_files))

        total_labels = np.sum(count.values())
        self.logger.info("Total Class distribution = {}".format(count))
        self.logger.info("Total labels = {}".format(total_labels))
        self.logger.info("Light = {}, Deep = {}, REM = {}, Wake = {}".format(count[2], count[1], count[3], count[4]))
        self.logger.info(
            "%Light = {}, %Deep = {}, %REM = {}, %Wake = {}".format((count[2]/float(total_labels))*100,
                                                                          (count[1]/float(total_labels))*100,
                                                                          (count[3]/float(total_labels))*100,
                                                                          (count[4]/float(total_labels))*100
                                                                          ))



    def get_class_weight(self, training_data):

        for npz_files in training_data[0:1]:
            datafile = np.load(npz_files)
            labels = datafile['Labels']

            weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
            return weights

    # def get_data(self):
    #     """
    #     Function to view and debug the generators
    #     Returns: Output generated from the generators
    #
    #     """
    #     training_files, validation_files = self.split_dataset()
    #     # training_samples_count = self.get_total_sample_count(training_files)
    #
    #     # self.logger.debug("Total training samples = {}".format(training_samples_count))
    #     return list(SequenceGenerator(training_files))