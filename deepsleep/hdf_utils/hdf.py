import numpy as np
import h5py as h5
import datetime as dt
from sklearn.preprocessing import StandardScaler
import csv
import itertools
import logging

#from scipy.signal import butter, lfilter
from tools import *
class HDF:
	
	def __init__(self, filename):
		self.h5_handle = h5.File(filename, "a")
		self.processed_dtype  = np.dtype([('Time', 'float32'),
                                          ('Heart Rate', 'float64'),
                                          ('Breathing Rate', 'float64'),
                                          ('Movement', 'int64'),
                                          ('Stages', 'int64'),
                                          ('SDNN', 'float64'),
                                          ('RMSSD', 'float64'),
                                          ('VLF1', 'float64'),
                                          ('LF1', 'float64'),
                                          ('HF1', 'float64'),
                                          ('Relative HF1', 'float64'),
                                          ('VLF2', 'float64'),
                                          ('LF2', 'float64'),
                                          ('HF2', 'float64'),
                                          ('Relative HF2', 'float64'),
                                          ('VLF3', 'float64'),
                                          ('LF3', 'float64'),
                                          ('HF3', 'float64'),
                                          ('Relative HF3', 'float64')
                                         ])
		self.snoring_dtype    = np.dtype([('Snoring Time', 'float32'),
                                          ('Duration', 'int64'),
                                          ('Frequency', 'S32')
                                         ])
		self.jj_dtype         = np.dtype([('J Time', 'float32')])
		self.zc_dtype         = np.dtype([('Zero Crossing', 'float32')])
		self.raw_dtype        = np.dtype([('RAW Time', 'float32'),
										  ('Piezo Value', 'int32')
										 ])
		self.filtered_dtype   = np.dtype([('Filtered Time', 'float32'),
										  ('Heart Signal', 'float64'),
										  ('Breathing Signal', 'float64'),
										  ('Movement Signal', 'float64')
										 ])

		self.validation_rpeaks_dtype = np.dtype([('Validation R Time', 'float32'),
												 ('Validation RR Intervals', 'float64'),
												])

		self.validation_dtype = np.dtype([('Validation Time', 'float32'),
										  ('Validation Heart Rate', 'float64'),
										  ('Validation Breathing Rate', 'float64'),
										  ('Validation Stages', 'int64'),
										  ('Validation SDNN', 'float64'),
										  ('Validation RMSSD', 'float64'),
										  ('Validation VLF1', 'float64'),
										  ('Validation LF1', 'float64'),
										  ('Validation HF1', 'float64'),
										  ('Validation Relative HF1', 'float64'),
										  ('Validation VLF2', 'float64'),
										  ('Validation LF2', 'float64'),
										  ('Validation HF2', 'float64'),
										  ('Validation Relative HF2', 'float64'),
										  ('Validation VLF3', 'float64'),
										  ('Validation LF3', 'float64'),
										  ('Validation HF3', 'float64'),
										  ('Validation Relative HF3', 'float64')
										 ])
		self.logger = logging.getLogger(__name__)

	def get_file_handle(self):
		return self.h5_handle

	def create_table(self, group_handle, label, c_dtype):
		# print "Creating Table:", label
		table_handle = group_handle.create_dataset(label,(0,),dtype=c_dtype,chunks=True,maxshape=(None,))
		return table_handle

	def create_subject_folder(self, subject_str):
		print "Creating Subject Folder:", subject_str
		file_handle    = self.h5_handle
		subject_handle = file_handle.create_group(subject_str)
		return subject_handle

	def create_date_folder(self, subject_handle, date_str):
		print "Creating Date Folder:", date_str
		date_handle = subject_handle.create_group(date_str)
		return date_handle

	def get_processed_folder(self, date_handle):

		date_keys = date_handle.keys()
		processed_str = "PROCESSED"
		if processed_str not in date_keys:
			print "Creating Processed Folder"
			processed_handle = date_handle.create_group(processed_str)
		else:
			processed_handle = date_handle[processed_str]
		return processed_handle

	def get_raw_folder(self, date_handle):

		date_keys = date_handle.keys()
		raw_str = "RAW"
		if raw_str not in date_keys:
			print "Creating Raw Folder"
			raw_handle = date_handle.create_group(raw_str)
		else:
			raw_handle = date_handle[raw_str]
		return raw_handle


	def get_filtered_folder(self, date_handle):

		date_keys = date_handle.keys()
		filtered_str = "FILTERED"
		if filtered_str not in date_keys:
			print "Creating Filtered Folder"
			filtered_handle = date_handle.create_group(filtered_str)
		else:
			filtered_handle = date_handle[filtered_str]
		return filtered_handle


	def get_validation_folder(self, date_handle):

		date_keys = date_handle.keys()
		validation_str = "VALIDATION"
		if validation_str not in date_keys:
			print "Creating Validation Folder"
			validation_handle = date_handle.create_group(validation_str)
		else:
			validation_handle = date_handle[validation_str]
		return validation_handle


	def get_subject_handle(self, subject_str):

		file_handle = self.h5_handle
		file_keys   = file_handle.keys()

		if subject_str not in file_keys:
			subject_handle = self.create_subject_folder(subject_str)
		else:
			subject_handle = file_handle[subject_str]

		return subject_handle

	def get_date_handle(self, subject_str, date_str):

		file_handle = self.h5_handle
		file_keys   = file_handle.keys()

		if subject_str not in file_keys:
			subject_handle = self.create_subject_folder(subject_str)
		else:
			subject_handle = file_handle[subject_str]

		subject_keys = subject_handle.keys()

		if date_str not in subject_keys:
			date_handle = self.create_date_folder(subject_handle, date_str)
		else:
			date_handle = subject_handle[date_str]

		return date_handle
	
	def get_table_handle(self, group_handle, label, c_dtype):

		group_handle_keys = group_handle.keys()

		if label not in group_handle_keys:
			table_handle = self.create_table(group_handle,label,c_dtype)
		else:
			table_handle = group_handle[label]

		return table_handle


	def write_list(self, processed_table_handle, column_name, data):
		#arr = np.ndarray(len(filteredHeartData), filtered_dtype)
		arr = processed_table_handle[:]
		arr[column_name] = data
		processed_table_handle[:] = arr
		return 1

	def append_processed_data(self, subject_str, date_str, time_list = [], heart_rate_list = [],
		breathing_rate_list = [], stages_list = [], movement_list = [], sdnn_list = [], rmssd_list = [], 
		lf1_list = [], lf2_list = [], lf3_list = [], hf1_list = [], hf2_list = [], hf3_list = [], rel_hf1_list = [], 
		rel_hf2_list = [], rel_hf3_list = [], vlf1_list = [], vlf2_list = [], vlf3_list = [], snoring_time_list = [], 
		snoring_duration_list = [], snoring_frequency_list = [], j_time_list = [], zero_crossings_list = [], call_number = 0):


		date_handle             = self.get_date_handle(subject_str, date_str)
		processed_handle        = self.get_processed_folder(date_handle)
		processed_table_handle  = self.get_table_handle(processed_handle, "PROCESSED_TABLE", self.processed_dtype)

		table_size              = processed_table_handle.shape[0]


		time_list           = processed_table_handle["Time"].tolist() + time_list
		heart_rate_list     = processed_table_handle["Heart Rate"].tolist() + heart_rate_list                 
		breathing_rate_list = processed_table_handle["Breathing Rate"].tolist() + breathing_rate_list                
		stages_list         = processed_table_handle["Stages"].tolist() + stages_list   
		movement_list       = processed_table_handle["Movement"].tolist() + movement_list               
		sdnn_list           = processed_table_handle["SDNN"].tolist() + sdnn_list                
		rmssd_list          = processed_table_handle["RMSSD"].tolist() + rmssd_list                 
		lf1_list            = processed_table_handle["LF1"].tolist() + lf1_list                 
		lf2_list            = processed_table_handle["LF2"].tolist() + lf2_list                 
		lf3_list            = processed_table_handle["LF3"].tolist() + lf3_list                 
		vlf2_list           = processed_table_handle["VLF1"].tolist() + vlf1_list                 
		vlf2_list           = processed_table_handle["VLF2"].tolist() + vlf2_list                 
		vlf3_list           = processed_table_handle["VLF3"].tolist() + vlf3_list                 
		hf1_list 			= processed_table_handle["HF1"].tolist() + hf1_list                 
		hf2_list 			= processed_table_handle["HF2"].tolist() + hf2_list                 
		hf3_list 			= processed_table_handle["HF3"].tolist() + hf3_list                 
		rel_hf1_list		= processed_table_handle["Relative HF1"].tolist() + rel_hf1_list
		rel_hf2_list 		= processed_table_handle["Relative HF2"].tolist() + rel_hf2_list
		rel_hf3_list 		= processed_table_handle["Relative HF3"].tolist() + rel_hf3_list


		snoring_table_handle = self.get_table_handle(processed_handle, "SNORING_TABLE", self.snoring_dtype)

		snoring_table_size = snoring_table_handle.shape[0]

		snoring_time_list           = snoring_table_handle["Snoring Time"].tolist() + snoring_time_list
		snoring_duration_list       = snoring_table_handle["Duration"].tolist() + snoring_duration_list
		snoring_frequency_list      = snoring_table_handle["Frequency"].tolist() + snoring_frequency_list

		jj_time_table_handle = self.get_table_handle(processed_handle, "JJ_TABLE", self.jj_dtype)

		jj_table_size = jj_time_table_handle.shape[0]

		j_time_list           = jj_time_table_handle["J Time"].tolist() + j_time_list
	
		zc_table_handle = self.get_table_handle(processed_handle, "ZERO_CROSSINGS", self.zc_dtype)

		zc_table_size = zc_table_handle.shape[0]

		zero_crossings_list   = zc_table_handle["Zero Crossing"].tolist() + zero_crossings_list
		
		response = self.set_processed_data(subject_str, date_str, time_list = time_list,
										   heart_rate_list = heart_rate_list, breathing_rate_list = breathing_rate_list,
										   stages_list = stages_list, movement_list = movement_list,
										   sdnn_list = sdnn_list, rmssd_list = rmssd_list, lf1_list = lf1_list,
										   lf2_list = lf2_list, lf3_list = lf3_list, hf1_list = hf1_list,
										   hf2_list = hf2_list, hf3_list = hf3_list, rel_hf1_list = rel_hf1_list,
										   rel_hf2_list = rel_hf2_list, rel_hf3_list = rel_hf3_list,
										   vlf1_list = vlf1_list, vlf2_list = vlf2_list, vlf3_list = vlf3_list,
										   snoring_time_list = snoring_time_list,
										   snoring_duration_list = snoring_duration_list,
										   snoring_frequency_list = snoring_frequency_list, j_time_list = j_time_list,
										   zero_crossings_list = zero_crossings_list, permission_flag = 1,
										   call_number = call_number)

		return response

	def pre_list_writing(self, data_list, table_handle, len_time_list, data_label, padding_flag, table_size):
		if len(data_list) != 0:
			if len(data_list) == len_time_list:
				if padding_flag == 1:
					for it in range(0,(table_size -len_time_list)):
						data_list.append(-1)		
				self.write_list(table_handle, data_label, data_list)
			else:
				print  data_label, " List not written, length doesn't match time list. Sad."
				pass

		return 1

	def set_processed_data(self, subject_str, date_str, time_list = [], heart_rate_list = [], breathing_rate_list = [],
						   stages_list = [], movement_list = [], sdnn_list = [], rmssd_list = [], lf1_list = [],
						   lf2_list = [], lf3_list = [], hf1_list = [], hf2_list = [], hf3_list = [], rel_hf1_list = [],
						   rel_hf2_list = [], rel_hf3_list = [], vlf1_list = [], vlf2_list = [], vlf3_list = [],
						   snoring_time_list = [], snoring_duration_list = [], snoring_frequency_list = [],
						   j_time_list = [], zero_crossings_list = [], permission_flag = 1, call_number = 0):

		if len(time_list) == 0 and len(snoring_time_list) == 0 and len(j_time_list) == 0 and len (zero_crossings_list) == 0:
			print "Time list cannot be empty!"
			return 0

		date_handle = self.get_date_handle(subject_str, date_str)
		processed_handle = self.get_processed_folder(date_handle)

		if len(time_list) != 0:
			processed_table_handle = self.get_table_handle(processed_handle, "PROCESSED_TABLE", self.processed_dtype)

			if call_number == 0:
				dateset_labels = "Time, Heart Rate, Breathing Rate, Movement, Stages, SDNN, RMSSD, VLF1, LF1, HF1, " \
								 "Relative HF1, VLF2, LF2, HF2, Relative HF2, VLF3, LF3, HF3, Relative HF3"
				self.set_dataset_attributes(processed_table_handle, dateset_labels)

			table_size             = processed_table_handle.shape[0]

			if table_size != 0 and permission_flag != 1:
				
				print "Existing Table's data length:", table_size
				print "Length of the new data;      ", len(time_list)
				print "Press 1 to continue, anything else to stop."
				response = raw_input()

				if response != "1":
					return 0
			
			padding_flag = 0

			if table_size > len(time_list):
				print "WARNING: Padding the new data with -1, It might affect the time integrity of old data."
				padding_flag = 1
			elif table_size < len(time_list):
				print "WARNING: Padding the old data with -1, It might affect the time integrity of old data."
				processed_table_handle.resize([len(time_list)])

			if padding_flag == 1:
				for it in range(0,(table_size - len(time_list))):
					time_list.append(-1)
			self.write_list(processed_table_handle, "Time", time_list)


			self.pre_list_writing(heart_rate_list, processed_table_handle, len(time_list), "Heart Rate", padding_flag, table_size)
			self.pre_list_writing(breathing_rate_list, processed_table_handle, len(time_list), "Breathing Rate", padding_flag, table_size)
			self.pre_list_writing(stages_list, processed_table_handle, len(time_list), "Stages", padding_flag, table_size)
			self.pre_list_writing(movement_list, processed_table_handle, len(time_list), "Movement", padding_flag, table_size)
			self.pre_list_writing(sdnn_list, processed_table_handle, len(time_list), "SDNN", padding_flag, table_size)
			self.pre_list_writing(rmssd_list, processed_table_handle, len(time_list), "RMSSD", padding_flag, table_size)
			self.pre_list_writing(lf1_list, processed_table_handle, len(time_list), "LF1", padding_flag, table_size)
			self.pre_list_writing(lf2_list, processed_table_handle, len(time_list), "LF2", padding_flag, table_size)
			self.pre_list_writing(lf3_list, processed_table_handle, len(time_list), "LF3", padding_flag, table_size)
			self.pre_list_writing(vlf1_list, processed_table_handle, len(time_list), "VLF1", padding_flag, table_size)
			self.pre_list_writing(vlf2_list, processed_table_handle, len(time_list), "VLF2", padding_flag, table_size)
			self.pre_list_writing(vlf3_list, processed_table_handle, len(time_list), "VLF3", padding_flag, table_size)
			self.pre_list_writing(hf1_list, processed_table_handle, len(time_list), "HF1", padding_flag, table_size)
			self.pre_list_writing(hf2_list, processed_table_handle, len(time_list), "HF2", padding_flag, table_size)
			self.pre_list_writing(hf3_list, processed_table_handle, len(time_list), "HF3", padding_flag, table_size)
			self.pre_list_writing(rel_hf1_list, processed_table_handle, len(time_list), "Relative HF1", padding_flag, table_size)
			self.pre_list_writing(rel_hf2_list, processed_table_handle, len(time_list), "Relative HF2", padding_flag, table_size)
			self.pre_list_writing(rel_hf3_list, processed_table_handle, len(time_list), "Relative HF3", padding_flag, table_size)


		if len(snoring_time_list) != 0:

			snoring_table_handle = self.get_table_handle(processed_handle, "SNORING_TABLE", self.snoring_dtype)
			if call_number == 0:
				dataset_labels = "Snoring Time, Duration, Frequency"
				self.set_dataset_attributes(snoring_table_handle, dataset_labels)

			snoring_table_size = snoring_table_handle.shape[0]
			
			if snoring_table_size != 0 and permission_flag != 1:
				
				print "Existing Snoring Table's data length:", snoring_table_size
				print "Length of the new Snoring data;      ", len(snoring_time_list)
				print "Press 1 to continue, anything else to stop."
				response = raw_input()

				if response != "1":
					return 0
			
			padding_flag = 0

			if snoring_table_size > len(snoring_time_list):
				print "WARNING: Padding the new data with -1, It might affect the time integrity of old data."
				padding_flag = 1
			elif snoring_table_size < len(snoring_time_list):
				print "WARNING: Padding the old data with -1, It might affect the time integrity of old data."
				snoring_table_handle.resize([len(snoring_time_list)])

			if padding_flag == 1:
				for it in range(0,(snoring_table_size - len(snoring_time_list))):
					snoring_time_list.append(-1)
			
			self.write_list(snoring_table_handle, "Snoring Time", snoring_time_list)			
			self.pre_list_writing(snoring_duration_list, snoring_table_handle, len(snoring_time_list), "Duration", padding_flag, snoring_table_size)
			self.pre_list_writing(snoring_frequency_list, snoring_table_handle, len(snoring_time_list), "Frequency", padding_flag, snoring_table_size)


		if len(j_time_list) != 0:
			
			jj_time_table_handle = self.get_table_handle(processed_handle, "JJ_TABLE", self.jj_dtype)
			if call_number == 0:
				dataset_labels = "J Time"
				self.set_dataset_attributes(jj_time_table_handle, dataset_labels)


			jj_table_size = jj_time_table_handle.shape[0]
			
			if jj_table_size != 0 and permission_flag != 1:
				
				print "Existing J Time Table's data length:", jj_table_size
				print "Length of the new J Time data;      ", len(j_time_list)
				print "Press 1 to continue, anything else to stop."
				response = raw_input()

				if response != "1":
					return 0
			


			jj_time_table_handle.resize([len(j_time_list)])
			
			self.write_list(jj_time_table_handle, "J Time", j_time_list)			


		if len(zero_crossings_list) != 0:
			
			zc_table_handle = self.get_table_handle(processed_handle, "ZERO_CROSSINGS", self.zc_dtype)
			if call_number == 0:
				dataset_labels = "Zero Crossing"
				self.set_dataset_attributes(zc_table_handle, dataset_labels)

			zc_table_size = zc_table_handle.shape[0]
			
			if zc_table_size != 0 and permission_flag != 1:
				
				print "Existing Zero Crossings Table's data length:", zc_table_size
				print "Length of the new Zero Xrossings data;      ", len(zero_crossings_list)
				print "Press 1 to continue, anything else to stop."
				response = raw_input()

				if response != "1":
					return 0
			


			zc_table_handle.resize([len(zero_crossings_list)])
			
			self.write_list(zc_table_handle, "Zero Crossing", zero_crossings_list)			


		return processed_handle

	def set_dataset_attributes(self, dataset_handle, dataset_labels):
		dataset_handle.attrs['column_name'] = dataset_labels


	def set_date_attributes(self, subject_str, date_str, attribute_label, attribute_value):
		date_handle             = self.get_date_handle(subject_str, date_str)
		date_handle.attrs[attribute_label]=attribute_value
		
	
	def set_raw_data(self, subject_str, date_str, epoch_number = 0, raw_time_list = [], piezo_value_list = []):
		if len(raw_time_list) == 0:
			print "RAW Time list cannot be empty!"
			return 0
		date_handle            = self.get_date_handle(subject_str, date_str)
		raw_handle             = self.get_raw_folder(date_handle)
		raw_data_label         = "RAW_Epoch_" + str(epoch_number).zfill(5)
		raw_table_handle       = self.get_table_handle(raw_handle, raw_data_label, self.raw_dtype)

		raw_table_size         = raw_table_handle.shape[0]
		padding_flag = 0
		raw_table_handle.resize([len(raw_time_list)])
		self.write_list(raw_table_handle, "RAW Time", raw_time_list)
		self.pre_list_writing(piezo_value_list, raw_table_handle, len(piezo_value_list), "Piezo Value", padding_flag, raw_table_size)

		return raw_handle

	
	def set_filtered_data(self, subject_str, date_str, epoch_number = 0, filtered_time_list = [], heart_value_list = [],
						  breathing_value_list = [], movement_value_list = []):
		if len(filtered_time_list) == 0:
			print "Filtered Time list cannot be empty!"
			return 0
		date_handle            = self.get_date_handle(subject_str, date_str)
		filtered_handle        = self.get_filtered_folder(date_handle)
		filtered_data_label    = "FILTERED_Epoch_" + str(epoch_number).zfill(5)
		filtered_table_handle  = self.get_table_handle(filtered_handle, filtered_data_label, self.filtered_dtype)

		filtered_table_size             = filtered_table_handle.shape[0]
		padding_flag = 0
		filtered_table_handle.resize([len(filtered_time_list)])

		self.write_list(filtered_table_handle, "Filtered Time", filtered_time_list)
		self.pre_list_writing(heart_value_list, filtered_table_handle, len(heart_value_list), "Heart Signal", padding_flag, filtered_table_size)
		self.pre_list_writing(breathing_value_list, filtered_table_handle, len(breathing_value_list), "Breathing Signal", padding_flag, filtered_table_size)
		self.pre_list_writing(movement_value_list, filtered_table_handle, len(movement_value_list), "Movement Signal", padding_flag, filtered_table_size)

		return filtered_handle


	def set_validation_data(self, subject_str, date_str, validation_r_time_list = [], validation_time_list = [],
							rr_interval_list = [], heart_rate_list = [], breathing_rate_list = [], stages_list = [],
							sdnn_list = [],	rmssd_list = [], vlf1_list = [], vlf2_list = [], vlf3_list = [],
							lf1_list = [], lf2_list = [], lf3_list = [], hf1_list = [], hf2_list = [], hf3_list = [],
							rel_hf1_list = [], rel_hf2_list = [], rel_hf3_list = [], permission_flag = 0):
		date_handle             = self.get_date_handle(subject_str, date_str)
		validation_handle       = self.get_validation_folder(date_handle)
		validation_table_handle = self.get_table_handle(validation_handle, "VALIDATION_RPEAKS_TABLE",
														self.validation_rpeaks_dtype)

		validation_table_size   = validation_table_handle.shape[0]

		if validation_table_size != 0 and permission_flag != 1:

			print "Existing Table's data length:", validation_table_size
			print "Length of the new data;      ", len(validation_r_time_list)
			print "Press 1 to continue, anything else to stop."
			response = raw_input()

			if response != "1":
				return 0

		padding_flag = 0

		if validation_table_size > len(validation_r_time_list):
			print "WARNING: Padding the new data with -1, It might affect the time integrity of old data."
			padding_flag = 1
		elif validation_table_size < len(validation_r_time_list):
			print "WARNING: Padding the old data with -1, It might affect the time integrity of old data."
			validation_table_handle.resize([len(validation_r_time_list)])

		if padding_flag == 1:
			for it in range(0,(validation_table_size - len(validation_r_time_list))):
				validation_r_time_list.append(-1)
		dateset_labels = "Validation R Time, Validation RR Intervals"
		self.set_dataset_attributes(validation_table_handle, dateset_labels)
		self.write_list(validation_table_handle, "Validation R Time", validation_r_time_list)
		self.pre_list_writing(rr_interval_list, validation_table_handle, len(validation_r_time_list),\
							  "Validation RR Intervals", padding_flag, validation_table_size)

		validation_table_handle = self.get_table_handle(validation_handle, "VALIDATION_TABLE",
														self.validation_dtype)
		validation_table_size = validation_table_handle.shape[0]
		if validation_table_size != 0 and permission_flag != 1:

			print "Existing Table's data length:", validation_table_size
			print "Length of the new data;      ", len(validation_r_time_list)
			print "Press 1 to continue, anything else to stop."
			response = raw_input()

			if response != "1":
				return 0

		padding_flag = 0

		if validation_table_size > len(validation_time_list):
			print "WARNING: Padding the new data with -1, It might affect the time integrity of old data."
			padding_flag = 1
		elif validation_table_size < len(validation_time_list):
			print "WARNING: Padding the old data with -1, It might affect the time integrity of old data."
			validation_table_handle.resize([len(validation_time_list)])

		if padding_flag == 1:
			for it in range(0,(validation_table_size - len(validation_r_time_list))):
				validation_r_time_list.append(-1)

		dateset_labels = "Validation Time, Validation Heart Rate, Validation Breathing Rate, Validation Stages, " \
						 "Validation SDNN, Validation RMSSD, Validation VLF1, Validation VLF2, Validation VLF3, " \
						 "Validation LF1, Validation LF2, Validation LF3, Validation HF1, Validation HF2, " \
						 "Validation HF3, Validation Relative HF1, Validation Relative HF2, Validation Relative HF3"

		self.set_dataset_attributes(validation_table_handle, dateset_labels)

		self.write_list(validation_table_handle, "Validation Time", validation_time_list)
		self.pre_list_writing(heart_rate_list, validation_table_handle, len(validation_time_list), \
							  "Validation Heart Rate", padding_flag, validation_table_size)
		self.pre_list_writing(breathing_rate_list, validation_table_handle, len(validation_time_list), \
							  "Validation Breathing Rate", padding_flag, validation_table_size)
		self.pre_list_writing(stages_list, validation_table_handle, len(validation_time_list), \
							  "Validation Stages", padding_flag, validation_table_size)
		#print sdnn_list
		self.pre_list_writing(sdnn_list, validation_table_handle, len(validation_time_list), \
							  "Validation SDNN", padding_flag, validation_table_size)
		self.pre_list_writing(rmssd_list, validation_table_handle, len(validation_time_list), \
							  "Validation RMSSD", padding_flag, validation_table_size)
		self.pre_list_writing(vlf1_list, validation_table_handle, len(validation_time_list), \
							  "Validation VLF1", padding_flag, validation_table_size)
		self.pre_list_writing(vlf2_list, validation_table_handle, len(validation_time_list), \
							  "Validation VLF2", padding_flag, validation_table_size)
		self.pre_list_writing(vlf3_list, validation_table_handle, len(validation_time_list), \
							  "Validation VLF3", padding_flag, validation_table_size)
		self.pre_list_writing(lf1_list, validation_table_handle, len(validation_time_list), \
							  "Validation LF1", padding_flag, validation_table_size)
		self.pre_list_writing(lf2_list, validation_table_handle, len(validation_time_list), \
							  "Validation LF2", padding_flag, validation_table_size)
		self.pre_list_writing(lf3_list, validation_table_handle, len(validation_time_list), \
							  "Validation LF3", padding_flag, validation_table_size)
		self.pre_list_writing(hf1_list, validation_table_handle, len(validation_time_list), \
							  "Validation HF1", padding_flag, validation_table_size)
		self.pre_list_writing(hf2_list, validation_table_handle, len(validation_time_list), \
							  "Validation HF2", padding_flag, validation_table_size)
		self.pre_list_writing(hf3_list, validation_table_handle, len(validation_time_list), \
							  "Validation HF3", padding_flag, validation_table_size)
		self.pre_list_writing(rel_hf1_list, validation_table_handle, len(validation_time_list), \
							  "Validation Relative HF1", padding_flag, validation_table_size)
		self.pre_list_writing(rel_hf2_list, validation_table_handle, len(validation_time_list), \
							  "Validation Relative HF2", padding_flag, validation_table_size)
		self.pre_list_writing(rel_hf3_list, validation_table_handle, len(validation_time_list), \
							  "Validation Relative HF3", padding_flag, validation_table_size)




		return validation_handle

	def get_processed_data(self, subject_str, date_str, column_name_values_for_processed = [],
						   column_name_values_for_snoring = [], start_time = 0, end_time = 0):
		date_handle            = self.get_date_handle(subject_str, date_str)
		fmt = '%Y-%m-%d %H:%M:%S'
		if start_time != 0 and end_time != 0:
			start_time = dt.datetime.strptime(start_time,fmt)
			end_time = dt.datetime.strptime(end_time,fmt)
		night_start_time       = self.get_night_start_time(date_handle)
		processed_handle       = self.get_processed_folder(date_handle)
		processed_handle_keys  = processed_handle.keys()
		processed_main_dictionary = {}
		processed_data = {}
		snoring_data = {}
		jj_values_data = {}
		zc_values_data = {}
		for table in processed_handle_keys:
			#print table
			if table == "PROCESSED_TABLE":
				processed_table_handle = self.get_table_handle(processed_handle, table, self.processed_dtype)
				processed_data = self.get_required_data(table_handle = processed_table_handle,
														column_name_values = column_name_values_for_processed,
														start_time = start_time, end_time = end_time,
														night_start_time = night_start_time, type_of_time = "Time")
				processed_main_dictionary[table] = processed_data

			elif table == "SNORING_TABLE":
				processed_table_handle = self.get_table_handle(processed_handle, table, self.snoring_dtype)
				snoring_data = self.get_required_data(table_handle = processed_table_handle,
													  column_name_values = column_name_values_for_snoring,
													  start_time = start_time, end_time = end_time,
													  night_start_time = night_start_time, type_of_time = "Snoring Time")
				processed_main_dictionary[table] = snoring_data

			elif table == "JJ_TABLE":
				processed_table_handle = self.get_table_handle(processed_handle, table, self.jj_dtype)
				jj_values_data = self.for_time_range(column_name = ['J Time'], start_time = start_time,
													 end_time = end_time, night_start_time = night_start_time,
													 table_handle = processed_table_handle, type_of_time = "J Time")
				jj_diff = np.diff(jj_values_data["J Time"])
				jj_values_data["JJ Diff"] = []
				for diff in jj_diff:
					jj_values_data["JJ Diff"].append(diff.total_seconds())

				processed_main_dictionary[table] = jj_values_data

			elif table == "ZERO_CROSSINGS":
				processed_table_handle = self.get_table_handle(processed_handle, table, self.zc_dtype)
				zc_values_data = self.for_time_range(column_name = ['Zero Crossing'],
													 start_time = start_time, end_time = end_time,
													 night_start_time = night_start_time,
													 table_handle = processed_table_handle,
													 type_of_time = "Zero Crossing")
				processed_main_dictionary[table] = zc_values_data

		return processed_main_dictionary

	def get_required_data(self, table_handle = None, column_name_values = [], night_start_time = None, start_time = 0,
						  end_time = 0, type_of_time = ""):
		data_dictionary = {}
		# print column_name_values
		try:
			if column_name_values == "All":
				# print "Here"
				column_name = table_handle.attrs['column_name'].split(", ")
				#column_name = ["Snoring Time"]
				data_dictionary = self.for_time_range(column_name = column_name, start_time = start_time,
													  end_time = end_time, night_start_time = night_start_time,
													  table_handle= table_handle, type_of_time = type_of_time)
			elif column_name_values == []:
				data_dictionary = {}
			else:
				data_dictionary = self.for_time_range(column_name = column_name_values, start_time = start_time ,
													  end_time = end_time, night_start_time = night_start_time,
													  table_handle = table_handle, type_of_time = type_of_time)
		except:
			import traceback
			print traceback.print_exc()
			pass
		return data_dictionary

	def get_time_list(self, time_list = [], night_start_time = None):
		j_time_list = []
		for time_values in time_list:
			j_time_list.append(night_start_time + dt.timedelta(seconds=time_values))
		return j_time_list

	def for_time_range(self, column_name=[], start_time=0, end_time=0, night_start_time=0, table_handle=None,
					   type_of_time = ""):
		data_dictionary = {}
		for column in column_name:
			data_dictionary[column] = []
			if start_time == 0:
				for time_range in range(len(table_handle[column])):
					if column == "Time" or column == "Snoring Time" or column == "J Time" or column == "Zero Crossing" \
							or column == "Validation R Time" or column == "Validation Time":
						data_dictionary[column].append(night_start_time + dt.timedelta(seconds=float(table_handle[column][time_range])))
					else:
						data_dictionary[column].append(table_handle[column][time_range])
			else:
				try:
					for time_range in range(len(table_handle[column])):
						row_time = night_start_time + dt.timedelta(seconds=float(table_handle[type_of_time][time_range]))
						if row_time >= start_time and row_time <= end_time:
							if column == "Time" or column == "Snoring Time" or column == "J Time" or \
											column == "Zero Crossing" or column == "Validation R Time" or \
											column == "Validation Time":
								data_dictionary[column].append(row_time)
							else:
								data_dictionary[column].append(table_handle[column][time_range])
				except:
					#print night_start_time, table_handle[type_of_time]
					import traceback
					print traceback.print_exc()
		return data_dictionary

	def get_raw_data(self, subject_str, date_str, start_time = 0, end_time = 0):
		date_handle = self.get_date_handle(subject_str, date_str)
		fmt = '%Y-%m-%d %H:%M:%S'
		if start_time != 0 and end_time != 0:
			start_time = dt.datetime.strptime(start_time, fmt)
			end_time = dt.datetime.strptime(end_time, fmt)
		night_start_time = self.get_night_start_time(date_handle)
		raw_handle = self.get_raw_folder(date_handle)
		raw_handle_keys = raw_handle.keys()
		raw_dictionary = {}
		raw_dictionary['RAW Time'] = []
		raw_dictionary['Piezo Value'] = []

		if start_time == 0:
			for key in range(len(raw_handle_keys)):
				try:
					raw_time_list, piezo_value_list = self.write_for_raw_values(raw_handle=raw_handle,
																				raw_handle_key=raw_handle_keys[key],
																				night_start_time = night_start_time)

					raw_dictionary['RAW Time'] += raw_time_list
					raw_dictionary['Piezo Value'] += piezo_value_list
				except:
					continue
		else:
			for key in range(len(raw_handle_keys)):
				try:
					dataset_time = night_start_time + dt.timedelta(seconds=key*30)
					if dataset_time >= start_time and dataset_time <= end_time:
						raw_time_list, piezo_value_list = self.write_for_raw_values(raw_handle = raw_handle,
																					raw_handle_key = raw_handle_keys[key],
																					night_start_time = night_start_time)
						raw_dictionary['RAW Time'] += raw_time_list
						raw_dictionary['Piezo Value'] += piezo_value_list
				except:
					import traceback
					print traceback.print_exc()
					continue
		return raw_dictionary

	def write_for_raw_values(self, raw_handle = None, raw_handle_key = "", night_start_time = None):
		raw_table_handle = self.get_table_handle(raw_handle, raw_handle_key, self.raw_dtype)
		raw_time_list = []
		for time_value in raw_table_handle['RAW Time']:
			raw_time_list.append(night_start_time + dt.timedelta(seconds=float(time_value)))
		return raw_time_list, raw_table_handle['Piezo Value'].tolist()

	def get_night_start_time(self, date_handle = None):
		fmt = '%Y-%m-%d %H:%M:%S'
		night_start_time = dt.datetime.strptime(date_handle.attrs["Recording_start_time"], fmt)
		return night_start_time

	def get_filtered_data(self, subject_str, date_str, start_time = 0, end_time = 0):
		date_handle = self.get_date_handle(subject_str, date_str)
		fmt = '%Y-%m-%d %H:%M:%S'
		if start_time != 0 and end_time != 0:
			start_time = dt.datetime.strptime(start_time, fmt)
			end_time = dt.datetime.strptime(end_time, fmt)
		night_start_time = self.get_night_start_time(date_handle)
		filtered_handle = self.get_filtered_folder(date_handle)
		filtered_handle_keys = filtered_handle.keys()
		# self.logger.debug("Keys : {}".format(filtered_handle_keys))
		filtered_dictionary = {}
		filtered_dictionary['Filtered Time'] = []
		filtered_dictionary['Heart Signal'] = []
		filtered_dictionary['Breathing Signal'] = []
		filtered_dictionary['Movement Signal'] = []
		if start_time == 0:
			for key in range(len(filtered_handle_keys)):
				filtered_time_list, filtered_heart_signal, filtered_breathing_signal, filtered_movement_signal = \
					self.write_for_filtered_values(filtered_handle=filtered_handle,
												   filtered_handle_key=filtered_handle_keys[key],
												   night_start_time = night_start_time,
												   current_epoch = key)
				filtered_dictionary['Filtered Time'] += filtered_time_list
				filtered_dictionary['Heart Signal'] += filtered_heart_signal
				filtered_dictionary['Breathing Signal'] += filtered_breathing_signal
				filtered_dictionary['Movement Signal'] += filtered_movement_signal

		else:
			for key in range(len(filtered_handle_keys)):
				dataset_time = night_start_time + dt.timedelta(seconds=key * 30)
				if dataset_time >= start_time and dataset_time <= end_time:
					filtered_time_list, filtered_heart_signal, filtered_breathing_signal, filtered_movement_signal = \
						self.write_for_filtered_values(filtered_handle=filtered_handle,
													   filtered_handle_key=filtered_handle_keys[key],
													   night_start_time = night_start_time,
													   current_epoch=key)
					filtered_dictionary['Filtered Time'] += filtered_time_list
					filtered_dictionary['Heart Signal'] += filtered_heart_signal
					filtered_dictionary['Breathing Signal'] += filtered_breathing_signal
					filtered_dictionary['Movement Signal'] += filtered_movement_signal
		return filtered_dictionary

	def write_for_filtered_values(self, filtered_handle = None, filtered_handle_key = "", night_start_time = None, current_epoch = None):
		filtered_table_handle = self.get_table_handle(filtered_handle, filtered_handle_key, self.filtered_dtype)
		filtered_time_list = []
		old_time_list = []
		#
		# self.logger.debug("Current epoch : {}".format(current_epoch))
		for time_value in filtered_table_handle['Filtered Time']:
			filtered_time_list.append(night_start_time + dt.timedelta(seconds=float(time_value)))

			# old_time_list.append(night_start_time + dt.timedelta(seconds=float(time_value)))

		# self.logger.debug("Time list : {}".format(filtered_time_list[0:5]))
		# self.logger.debug("Old Time List : {}".format(old_time_list[0:5]))
		return filtered_time_list, filtered_table_handle['Heart Signal'].tolist(), \
			   filtered_table_handle['Breathing Signal'].tolist(), filtered_table_handle['Movement Signal'].tolist()


	def get_validation_data(self, subject_str, date_str, column_name_values_for_validation = [],
							start_time = 0, end_time = 0):
		# print "Validation"
		date_handle = self.get_date_handle(subject_str, date_str)
		fmt = '%Y-%m-%d %H:%M:%S'
		if start_time != 0 and end_time != 0:
			start_time = dt.datetime.strptime(start_time, fmt)
			end_time = dt.datetime.strptime(end_time, fmt)
		night_start_time = self.get_night_start_time(date_handle)
		validation_handle = self.get_validation_folder(date_handle)
		validation_handle_keys = validation_handle.keys()
		validation_dictionary = {}
		validation_dictionary["VALIDATION_RPEAKS_TABLE"] = []
		validation_dictionary["VALIDATION_TABLE"] = []

		for table in validation_handle_keys:
			if table == "VALIDATION_RPEAKS_TABLE":
				validation_table_handle = self.get_table_handle(validation_handle, table, self.validation_rpeaks_dtype)
				validation_r_data = self.get_required_data(table_handle=validation_table_handle,
													  column_name_values="All",
													  start_time=start_time, end_time=end_time,
													  night_start_time=night_start_time, type_of_time="Validation R Time")
				validation_dictionary[table] = validation_r_data

			elif table == "VALIDATION_TABLE":
				validation_table_handle = self.get_table_handle(validation_handle, table, self.validation_dtype)
				validation_data = self.get_required_data(table_handle=validation_table_handle,
														   column_name_values=column_name_values_for_validation,
														   start_time=start_time, end_time=end_time,
														   night_start_time=night_start_time,
														   type_of_time="Validation Time")
				validation_dictionary[table] = validation_data
		# print validation_dictionary["VALIDATION_TABLE"].keys()
		return validation_dictionary







	def get_trend_lines(self,time_list, rel_hf1, rel_hf2, rel_hf3, hf1, hf2, hf3, lf1, lf2, lf3):

		all_lf_hf = []
		all_relhf = []

		for iterator in range(0,len(time_list)):
			all_lf_hf.append(((lf1[iterator]/hf1[iterator])+(lf2[iterator]/hf2[iterator])+(lf3[iterator]/hf2[iterator]))/3)
			all_relhf.append((rel_hf1[iterator]+rel_hf2[iterator]+rel_hf3[iterator])/3)


		arr_all_lf_hf = np.array(all_lf_hf)
		arr_all_relhf = np.array(all_relhf)
		arr_all_lf_hf = arr_all_lf_hf.reshape(-1,1)
		arr_all_relhf = arr_all_relhf.reshape(-1,1)
		arr_all_lf_hf = StandardScaler().fit_transform(arr_all_lf_hf)
		arr_all_relhf = StandardScaler().fit_transform(arr_all_relhf)
		
		#all_time_list  = []
		#for i in time_list:
		#	all_time_list.append(dt.datetime.strptime(i,"%Y-%m-%d %H:%M:%S"))

		all_time_list = time_list
		all_lf_hf_sg = []
		all_relhf_sg = []

		for val in arr_all_lf_hf:
			all_lf_hf_sg.append(val[0])

		for val in arr_all_relhf:
			all_relhf_sg.append(val[0])

		
		return all_lf_hf_sg, all_relhf_sg

	def get_snoring_data(self, subject_str, date_str, snoring_limit_for_classification):

		processed_main_dictionary = self.get_processed_data(subject_str, date_str, column_name_values_for_processed = "", column_name_values_for_snoring = ["Snoring Time"])

		snoring_main_dictionary_handle = processed_main_dictionary["SNORING_TABLE"]

		snoring_time = snoring_main_dictionary_handle["Snoring Time"]
		
		snoring_rate_time_bl  = []
		snoring_rate_bl = []
		l1 = []
		snoring_rate_time_ab = []
		snoring_rate_ab = []
		l2 = []

		previous_min = -1

		sval = 1
		for snoring_itr in range(0,len(snoring_time)):

				stri    = dt.datetime.strftime(snoring_time[snoring_itr],'%Y-%m-%d %H:%M:%S')
				print stri
				snr_str =  stri[11:16]
				print snr_str
				if previous_min == -1:
					previous_min = snr_str

				if snr_str == previous_min:
					sval = sval + 1
				else:
					if sval < snoring_limit_for_classification and sval > 2:		
						snoring_rate_time_bl.append((snoring_time[snoring_itr-1].replace(second=0)))
						snoring_rate_bl.append(sval)
						l1.append(1)
					elif sval > 2:		
						snoring_rate_time_ab.append((snoring_time[snoring_itr-1].replace(second=0)))
						snoring_rate_ab.append(sval)
						l2.append(1)
					previous_min = snr_str
					sval = 1


		return snoring_rate_time_bl, snoring_rate_bl, snoring_rate_time_ab, snoring_rate_ab


	def get_movement_data(self, subject_str, date_str):

		movement_time_list = []

		processed_main_dictionary = self.get_processed_data(subject_str, date_str, column_name_values_for_processed = "All")
		
		processed_main_dictionary_handle = processed_main_dictionary["PROCESSED_TABLE"]

		movement_time = processed_main_dictionary_handle["Time"]

		movement = processed_main_dictionary_handle["Movement"]

		for epoch_itr in range(0,len(movement_time)):

			movement_dec = movement[epoch_itr]

			bin_str = '{0:b}'.format(movement_dec)

			bin_str = bin_str.zfill(30)

			for char_itr in range(0,len(bin_str)):

				if bin_str[char_itr] == '1':
					movement_time_list.append(movement_time[epoch_itr]+dt.timedelta(seconds=char_itr))
				

		return movement_time_list

	def get_psg_stages(self, subject_str, date_str):

		validation_main_dictionary = self.get_validation_data(subject_str, date_str, column_name_values_for_validation = "All")
		
		validation_main_dictionary_handle = validation_main_dictionary["VALIDATION_TABLE"]

		validation_time_list = validation_main_dictionary_handle["Validation Time"]

		validation_stages    = validation_main_dictionary_handle["Validation Stages"]

		return validation_time_list, validation_stages


	def get_ecg_heart_rate(self, subject_str, date_str):

		validation_main_dictionary = self.get_validation_data(subject_str, date_str, column_name_values_for_validation = "All")
		
		validation_main_dictionary_handle = validation_main_dictionary["VALIDATION_TABLE"]

		validation_time_list = validation_main_dictionary_handle["Validation Time"]

		validation_heart_rate_list = validation_main_dictionary_handle["Validation Heart Rate"]


		#validation_time_list_updated = []
		#validation_heart_rate_list_updated = []

		#for hr_itr in range(0,len(validation_time_list)):
		#	if validation_heart_rate_list[hr_itr] > 30 and validation_heart_rate_list[hr_itr] < 90:
		#		validation_time_list_updated.append(validation_time_list[hr_itr])
		#		validation_heart_rate_list_updated.append(validation_heart_rate_list[hr_itr])	


		return validation_time_list, validation_heart_rate_list

	def get_dozee_heart_rate(self, subject_str, date_str):

		processed_main_dictionary = self.get_processed_data(subject_str, date_str, column_name_values_for_processed = "All")
		
		processed_main_dictionary_handle = processed_main_dictionary["PROCESSED_TABLE"]

		processed_time = processed_main_dictionary_handle["Time"]

		heart_rate_list = processed_main_dictionary_handle["Heart Rate"]


		#validation_time_list_updated = []
		#validation_heart_rate_list_updated = []

		#for hr_itr in range(0,len(validation_time_list)):
		#	if validation_heart_rate_list[hr_itr] > 30 and validation_heart_rate_list[hr_itr] < 90:
		#		validation_time_list_updated.append(validation_time_list[hr_itr])
		#			validation_heart_rate_list_updated.append(validation_heart_rate_list[hr_itr])	


		return processed_time, heart_rate_list


	def get_dozee_breathing_rate(self, subject_str, date_str):

		processed_main_dictionary = self.get_processed_data(subject_str, date_str, column_name_values_for_processed = "All")
		
		processed_main_dictionary_handle = processed_main_dictionary["PROCESSED_TABLE"]

		breathing_time_list = processed_main_dictionary_handle["Time"]

		dozee_breathing_rate_list = processed_main_dictionary_handle["Breathing Rate"]

		#t1 = dt.datetime.strptime("2017-11-03 23:46:00",'%Y-%m-%d %H:%M:%S')
		t1 = dt.datetime.strptime("2017-11-03 23:51:00",'%Y-%m-%d %H:%M:%S')

		breathing_time_list_updated = []
		dozee_breathing_rate_list_updated = []

		for br_itr in range(0,len(breathing_time_list)):
			if dozee_breathing_rate_list[br_itr] > 6 and dozee_breathing_rate_list[br_itr] < 45:
				if breathing_time_list[br_itr] > t1:
					breathing_time_list_updated.append(breathing_time_list[br_itr])
					dozee_breathing_rate_list_updated.append(dozee_breathing_rate_list[br_itr])	


		return breathing_time_list_updated, dozee_breathing_rate_list_updated


	def get_subject_date_str(self):

		#[{"SUBJECT":"Subject20","DATE":"05112017"}]
		out_list = []
		file_handle = self.h5_handle
		file_keys   = file_handle.keys()		
		
		for subject in file_keys:
			subject_handle = file_handle[subject]
			subject_keys = subject_handle.keys()
			for date in subject_keys:
				temp_dict  ={}
				temp_dict["SUBJECT"] = subject
				temp_dict["DATE"] = date
				out_list.append(temp_dict)

		return out_list

	def get_ecg_scaled_lf_hf_rel_hf(self, subject_str, date_str, out_file):

		validation_main_dictionary = self.get_validation_data(subject_str, date_str, column_name_values_for_validation = "All")
		
		validation_main_dictionary_handle = validation_main_dictionary["VALIDATION_TABLE"]

		validation_time_list = validation_main_dictionary_handle["Validation Time"]

		lf1 = validation_main_dictionary_handle["Validation LF1"]
		lf2 = validation_main_dictionary_handle["Validation LF2"]
		lf3 = validation_main_dictionary_handle["Validation LF3"]
		hf1 = validation_main_dictionary_handle["Validation HF1"]
		hf2 = validation_main_dictionary_handle["Validation HF2"]
		hf3 = validation_main_dictionary_handle["Validation HF3"]
		rel_hf1 = validation_main_dictionary_handle["Validation Relative HF1"]
		rel_hf2 = validation_main_dictionary_handle["Validation Relative HF2"]
		rel_hf3 = validation_main_dictionary_handle["Validation Relative HF3"]
		
		all_lf_hf_sg, all_relhf_sg = self.get_trend_lines(validation_time_list, rel_hf1, rel_hf2, rel_hf3, hf1, hf2, hf3, lf1, lf2, lf3)

		sampling_rate = int(len(validation_time_list)/20)
		lf_hf, _, _ = filter_signal(signal=all_lf_hf_sg,
	                                          ftype='butter',
	                                          band='bandpass',
	                                          order=3,
	                                          frequency=[0,0.5],
	                                          sampling_rate=sampling_rate)

		rel_hf, _, _ = filter_signal(signal=all_relhf_sg,
	                                          ftype='butter',
	                                          band='bandpass',
	                                          order=3,
	                                          frequency=[0,0.5],
	                                          sampling_rate=sampling_rate)

		return validation_time_list, lf_hf, rel_hf

	def write_report_csv(self, subject_str, date_str, out_file):

		snoring_rate_time_mild, snoring_rate_mild, snoring_rate_time_heavy, snoring_rate_heavy = self.get_snoring_data(subject_str, date_str,7)	

		movement_time_list = self.get_movement_data(subject_str, date_str)

		validation_time_list, validation_stages = self.get_psg_stages(subject_str, date_str)

		validation_heart_time_list, validation_heart_rate_list = self.get_ecg_heart_rate(subject_str, date_str)

		breathing_time_list_updated, dozee_breathing_rate_list_updated = self.get_dozee_breathing_rate(subject_str, date_str)

		validation_time_list, lf_hf, rel_hf = self.get_ecg_scaled_lf_hf_rel_hf( subject_str, date_str, out_file)
		
		recovery_time_list, recovery_list = self.get_ecg_recovery(subject_str, date_str, out_file)


		d = {"snoring_rate_time_mild": snoring_rate_time_mild, "snoring_rate_time_heavy": snoring_rate_time_heavy, 
			"movement_time_list": movement_time_list, "validation_time_list": validation_time_list, "validation_stages":validation_stages,
			"validation_heart_time_list": validation_heart_time_list, "validation_heart_rate_list": validation_heart_rate_list,
			"breathing_time_list_updated": breathing_time_list_updated, "dozee_breathing_rate_list_updated": dozee_breathing_rate_list_updated,
			"recovery_time_list" : recovery_time_list, "recovery_list": recovery_list, "validation_hrv_time_list": validation_time_list, "lfhf": lf_hf,
			"rel_hf": rel_hf}
		with open(out_file, "wb") as outfile:
			writer = csv.writer(outfile)
			writer.writerow(d.keys())
			writer.writerows(itertools.izip_longest(*d.values()))


	def get_proceesed_recovery(self, subject_str, date_str):
		date_handle = self.get_date_handle(subject_str, date_str)
		night_start_time = self.get_night_start_time(date_handle)
		processed_main_dictionary = self.get_processed_data(subject_str=subject_str, date_str=date_str,
															column_name_values_for_processed=['Time', 'SDNN'])
		base_value = self.find_base_value(night_start_time, processed_main_dictionary['PROCESSED_TABLE']['Time'],
										processed_main_dictionary['PROCESSED_TABLE']['SDNN'])

		point_system = self.find_recovery_score(base_value,	processed_main_dictionary['PROCESSED_TABLE']['SDNN'])

		recovery_dictionary = {}
		recovery_dictionary['Recovery Time'] = processed_main_dictionary['PROCESSED_TABLE']['Time']
		recovery_dictionary['Recovery Values'] = point_system

		return recovery_dictionary

	def get_ecg_recovery(self, subject_str, date_str):
		date_handle = self.get_date_handle(subject_str, date_str)
		night_start_time = self.get_night_start_time(date_handle)
		validation_main_dictionary = self.get_validation_data(subject_str=subject_str, date_str=date_str,
															  column_name_values_for_validation=['Validation Time',
																								 'Validation SDNN'])
		base_value = self.find_base_value(night_start_time,
										  validation_main_dictionary['VALIDATION_TABLE']['Validation Time'],
										  validation_main_dictionary['VALIDATION_TABLE']['Validation SDNN'])

		point_system = self.find_recovery_score(base_value,
												validation_main_dictionary['VALIDATION_TABLE']['Validation SDNN'])

		validation_recovery_dictionary = {}
		validation_recovery_dictionary['Validation Recovery Time'] = validation_main_dictionary['VALIDATION_TABLE']['Validation Time']
		validation_recovery_dictionary['Validation Recovery Values'] = point_system

		return validation_recovery_dictionary

	def find_base_value(self, startTime, timeValue, recoveryParameter):
		firstHour = []
		for dataValues in range(len(timeValue)):
			if timeValue[dataValues] < startTime + dt.timedelta(hours=1):
				if recoveryParameter[dataValues] < 60 and recoveryParameter[dataValues] != -1 and \
						recoveryParameter[dataValues] != -2:
					firstHour.append(recoveryParameter[dataValues])

		baseValue = np.mean(firstHour)
		if np.isnan(baseValue):
			baseValue = 60
		return baseValue

	def find_recovery_score(self, baseValue, recoveryParameter):
		upperRange = baseValue + .1 * baseValue
		lowerRange = baseValue - .2 * baseValue
		upperLimit = baseValue + .2 * baseValue
		lowerLimit = baseValue - .4 * baseValue

		zeroScore = 0
		oneScore = 0
		twoScore = 0
		threeScore = 0
		fourScore = 0
		fiveScore = 0

		completeScore = []
		totalIterations = 0

		if len(recoveryParameter) > 840:
			numberOfEpochs = len(recoveryParameter)
		else:
			numberOfEpochs = 840

		for dataValue in range(len(recoveryParameter)):
			if recoveryParameter[dataValue] < 150:
				if dataValue == 0 or completeScore == []:
					prevVal = 0.0
				else:
					prevVal = float(completeScore[-1])

				if recoveryParameter[dataValue] < lowerLimit:
					zeroScore += 1
					totalIterations += 1
					completeScore.append((0 + prevVal))

				elif recoveryParameter[dataValue] >= lowerLimit and recoveryParameter[dataValue] < lowerRange:
					oneScore += 1
					totalIterations += 1
					completeScore.append((1 + prevVal))

				elif recoveryParameter[dataValue] >= lowerRange and recoveryParameter[dataValue] < baseValue:
					twoScore += 1
					totalIterations += 1
					completeScore.append((2 + prevVal))

				elif recoveryParameter[dataValue] >= baseValue and recoveryParameter[dataValue] < upperRange:
					threeScore += 1
					totalIterations += 1
					completeScore.append((3 + prevVal))

				elif recoveryParameter[dataValue] >= upperRange and recoveryParameter[dataValue] < upperLimit:
					fourScore += 1
					totalIterations += 1
					completeScore.append((4 + prevVal))

				elif recoveryParameter[dataValue] >= upperLimit:
					fiveScore += 1
					totalIterations += 1
					completeScore.append((5 + prevVal))

			elif completeScore == []:
				completeScore.append(0)
			else:
				completeScore.append(completeScore[-1])

		pointSystem = []
		for point in completeScore:
			pointSystem.append(point / numberOfEpochs)

		return pointSystem