import os, glob
import logging
import numpy as np
import h5py as h5
from collections import Counter

from utils import ResampleDataset
from preprocess import PreProcessData
from keras.preprocessing import sequence


class PrepareDataset(object):
    """
    Extract data from HDF files and store it Epoch-based.
    This method is more suitable to handle different sequence lengths by using `pad_sequences` method from Keras.
    """
    def __init__(self, hdf_file_path, seq_len, batch_size, n_classes):
        self.filepath = hdf_file_path
        self.ROOT = os.getcwd()
        self.STORE_DIR = os.path.join(self.ROOT, "dataset")

        self.logger = logging.getLogger(__name__)

        self.seq_len = seq_len

        self.preprocess = PreProcessData(seq_len=seq_len, batch_size=batch_size, n_classes=n_classes)

    def open_hdf(self):
        self.hdf_object = h5.File(self.filepath, 'a')
        #self.logger.info("File: {}".format(self.hdf_object))
        print "File : {}".format(self.hdf_object)
        masterFileKeys = self.hdf_object.keys()

        #self.logger.debug("MasterFileKeys : {}".format(masterFileKeys))
        print "MasterFileKeys : {}".format(masterFileKeys)
        # return subject_metadata, file_handle
        return masterFileKeys

    def __get_samplingrate(self, filtered_time, filtered_hs):
        total_samples = len(filtered_hs)
        last_timestamp = filtered_time[-1]
        first_timestamp = filtered_time[0]
        total_seconds = last_timestamp - first_timestamp
        sampling_rate = total_samples / total_seconds

        return sampling_rate

    def get_filtered_epoch(self, date_handle):

        filteredGroupHandle = date_handle['FILTERED']
        filteredGroupKeys = filteredGroupHandle.keys()

        heart_signal = []
        timestamp = []
        sample_rate_list = []
        for epoch_count, filteredValue in enumerate(filteredGroupKeys):
            heart_epoch_data = filteredGroupHandle[filteredValue]['Heart Signal']
            time_epoch_data = filteredGroupHandle[filteredValue]['Filtered Time']
            sample_rate_epoch = self.__get_samplingrate(time_epoch_data, heart_epoch_data)

            sample_rate_list.append(sample_rate_epoch)
            heart_signal.append(heart_epoch_data)
            timestamp.append(time_epoch_data)

        return np.asarray(timestamp), np.asarray(heart_signal), np.asarray(sample_rate_list)

    def get_labels(self, date_handle):

        validationGroupHandle = date_handle['VALIDATION']

        validation_table_group = validationGroupHandle["VALIDATION_TABLE"]
        labels = validation_table_group['Validation Stages']
        validation_timestamps = validation_table_group['Validation Time']

        for ind, val in enumerate(labels):
            if labels[ind] < 0:
                labels[ind] = 4

        return validation_timestamps, labels

    def _preprocessor(self, heart_signal, labels, sampling_rate):

        # heart_signal, labels, sampling_rate = self.preprocess.load_data(npz_file)
        # self.logger.debug("Heart signal shape = {}".format(heart_signal.shape))

        # Standardize the heart signal for every 30 seconds epoch
        self.logger.info("Standardizing signals...")
        heart_signal = self.preprocess.standardize_data(heart_signal)
        self.logger.debug("Heart shape after standardization = {}".format(heart_signal.shape))

        # Create sequences
        self.logger.info("Creating and Padding sequences...")
        heart_signal_seq, labels_seq = self.preprocess.create_sequences(heart_signal, labels, sampling_rate=sampling_rate)
        self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))


        # Pad sequences to get uniform sequence length
        heart_signal_seq = sequence.pad_sequences(heart_signal_seq, maxlen=self.seq_len, dtype='float32',
                                                  padding='post',
                                                  truncating='post')
        self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))

        # Convert labels to categorical format
        # self.logger.info("Converting labels to categorical...")
        # labels_categorical = self.preprocess.convert_to_categorical(labels_seq)

        self.logger.debug("Shape = {}, {}".format(heart_signal_seq.shape, labels_seq.shape))

        return heart_signal_seq, labels_seq

    def prep_dataset(self):
        subject_key = self.open_hdf()

        for subject in subject_key:
            subjectGroupHandle = self.hdf_object[subject]
            subjectGroupKeys = subjectGroupHandle.keys()

            for dateValue in subjectGroupKeys:
                dateGroupHandle = subjectGroupHandle[dateValue]

                filtered_ts, filtered_hs, sampling_rate_list = self.get_filtered_epoch(date_handle=dateGroupHandle)
                validation_ts, labels = self.get_labels(date_handle=dateGroupHandle)

                filtered_hs, labels = self._preprocessor(heart_signal=filtered_hs, labels=labels, sampling_rate=sampling_rate_list)

                heart_flat_list = [item for sublist in filtered_hs for item in sublist]

                # epochs_to_consider = np.minimum(len(filtered_hs), len(labels))
                # filtered_hs = filtered_hs[:epochs_to_consider]
                # labels = labels[:epochs_to_consider]

                assert len(filtered_hs) == len(labels)

                filename = subject + "_" + dateValue

                save_dict = {
                    "Timestamp": filtered_ts,
                    "Filtered_Heart": filtered_hs,
                    "Labels": np.asarray(labels),
                    "Sampling_rate": sampling_rate_list,
                    "Total_samples": len(heart_flat_list),
                    "Total_epochs": len(filtered_hs),
                    "Subject_name": subject,
                    "date": dateValue
                }

                print "Saving {} to NPZ file ...".format(filename)
                np.savez(os.path.join(self.STORE_DIR, filename), **save_dict)

    def test_npz(self, npz_files):
        datafile = np.load(npz_files)
        print datafile.keys()

        print datafile['Timestamp'][:2]
        print datafile["Filtered_Heart"][:2]
        print datafile["Labels"][:2]
        print datafile["Total_samples"]
        print datafile["Total_epochs"]
        print datafile["Sampling_rate"]
        print datafile["Subject_name"]
        print datafile["date"]

class PreparePretrainDataset(object):
    def __init__(self, seq_len, batch_size, n_classes, sampling_mode = 'random_over_sample'):
        self.sampling_mode = sampling_mode
        self.ROOT = os.getcwd()
        self.STORE_DIR = os.path.join(self.ROOT, "pretrain_dataset")
        self.DATASET_DIR = os.path.join(self.ROOT, 'dataset')

        self.logger = logging.getLogger(__name__)

        self.seq_len = seq_len

        self.preprocess = PreProcessData(seq_len=seq_len, batch_size=batch_size, n_classes=n_classes)

    def _preprocessor(self, heart_signal, labels, sampling_rate):

        # heart_signal, labels, sampling_rate = self.preprocess.load_data(npz_file)
        # self.logger.debug("Heart signal shape = {}".format(heart_signal.shape))

        # Standardize the heart signal for every 30 seconds epoch
        self.logger.info("Standardizing signals...")
        heart_signal = self.preprocess.standardize_data(heart_signal)
        self.logger.debug("Heart shape after standardization = {}".format(heart_signal.shape))

        # Create sequences
        self.logger.info("Creating and Padding sequences...")
        heart_signal_seq, labels_seq = self.preprocess.create_sequences(heart_signal, labels, sampling_rate=sampling_rate)
        self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))


        # Pad sequences to get uniform sequence length
        heart_signal_seq = sequence.pad_sequences(heart_signal_seq, maxlen=self.seq_len, dtype='float32',
                                                  padding='post',
                                                  truncating='post')
        self.logger.debug("Heart signal shape = {}".format(heart_signal_seq.shape))

        # Convert labels to categorical format
        # self.logger.info("Converting labels to categorical...")
        # labels_categorical = self.preprocess.convert_to_categorical(labels_seq)

        # Add extra dimension to suit the requirements for LSTM & CNN
        heart_signal_seq = np.expand_dims(heart_signal_seq, 2)


        self.logger.debug("Shape = {}, {}".format(heart_signal_seq.shape, labels_seq.shape))

        return heart_signal_seq, labels_seq

    def store_pretrain_files(self):

        resampler = ResampleDataset(sampling_mode=self.sampling_mode)
        for npz_file in glob.glob(self.DATASET_DIR + "/*.npz"):
            filename = npz_file.split('/')[-1]
            print "Loading {} for pretraining ...".format(filename)

            datafile = np.load(npz_file)
            heart_signal = datafile['Filtered_Heart']
            labels = datafile['Labels']
            sampling_rate = datafile["Sampling_rate"]
            subjects = datafile['Subject_name']
            dates = datafile['date']

            self.logger.info("Distribution before balancing = {}".format(Counter(labels)))
            print type(heart_signal)

            # Resample dataset to balance classes
            heart_signal_res, labels_res = resampler.balance_classes(heart_signal=heart_signal, labels=labels)

            heart_signal_res, labels_res = self._preprocessor(heart_signal_res, labels_res, sampling_rate)
            self.logger.info("Distribution after balancing = {}".format(Counter(labels_res)))

            # Save as .npz file
            save_dict = {
                "Filtered_Heart": np.asarray(heart_signal_res),
                "Labels": np.asarray(labels_res),
                "Sampling_rate": sampling_rate,
                "Total_samples": len(heart_signal_res),
                "Subject_name": subjects,
                "date": dates
            }

            filename = str(subjects)+"_"+str(dates)
            print "Saving pretraining dataset ..."
            print "Saving {} to NPZ file ...".format(filename)
            self.logger.info("Writing {} into NPY".format(filename))

            np.savez(os.path.join(self.STORE_DIR, filename), **save_dict)