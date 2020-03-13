from hdf import HDF
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def compute_accuracy_of_one_night(hdf_obj, subject_label, date_label, to_plot = 0):

	#get ech heart
	validation_time_list, validation_heart_rate_list = hdf_obj.get_ecg_heart_rate(subject_label, date_label)

	#get dozee herat
	processed_time, heart_rate_list = hdf_obj.get_dozee_heart_rate(subject_label, date_label)


	processed_time_plot_corr = []
	heart_rate_list_plot_corr = []
	validation_time_list_plot_corr = [] 
	validation_heart_rate_list_plot_corr = [] 
	
	ecg_avg = np.average(validation_heart_rate_list)
	match_count = 0
	cumulative_error = 0
	#compute accuracy
	for dozee_itr in range(0,len(processed_time)):

		if heart_rate_list[dozee_itr] != -1 and  heart_rate_list[dozee_itr] != -2:
				ecg_index = -1
				for ecg_itr in range(0,len(validation_time_list)):
					if(processed_time[dozee_itr] == validation_time_list[ecg_itr]):
						ecg_index = ecg_itr
						break

				#ecg_index        = validation_time_list.index(processed_time[dozee_itr])
				#ecg_index        = dozee_itr
				
				if ecg_index != -1:	
					relative_error   = abs(heart_rate_list[dozee_itr]-validation_heart_rate_list[ecg_index])/validation_heart_rate_list[ecg_index] 
					cumulative_error = cumulative_error + relative_error
					match_count      = match_count + 1

					processed_time_plot_corr.append(processed_time[dozee_itr])
					heart_rate_list_plot_corr.append(heart_rate_list[dozee_itr])
					validation_time_list_plot_corr.append(validation_time_list[ecg_index]) 
					validation_heart_rate_list_plot_corr.append(validation_heart_rate_list[ecg_index])



	if match_count != 0:
		abs_rel_err = (cumulative_error * 100)/match_count
		accuracy = 100 - abs_rel_err


		print "Accuracy: ",accuracy
		correlation = np.corrcoef(validation_heart_rate_list_plot_corr,heart_rate_list_plot_corr)
		corr_value = correlation[0][1]
		print "Correlation: ",corr_value
		print 'Match Count', match_count
		print 'Total length', len(heart_rate_list)
		print 'Detection Rate', match_count / len(heart_rate_list)

	#plot
	if to_plot == 1:
		plt.figure()
		plt.subplot(1,1,1)
		plt.plot(validation_time_list_plot_corr,validation_heart_rate_list_plot_corr, label ="ECG")
		plt.plot(processed_time_plot_corr,heart_rate_list_plot_corr, label ="DOZEE")
		plt.show()

		plt.figure()
		plt.subplot(1,1,1)
		plt.plot(validation_heart_rate_list_plot_corr,heart_rate_list_plot_corr,'.')
		plt.show()
		
	return accuracy, corr_value

def compute_acuuracy(filename, subject_name_date_list = [],to_plot=0):

	out_dict_list = []
	hdf_obj = HDF(filename)

	if subject_name_date_list == "ALL":
		subject_name_date_list = hdf_obj.get_subject_date_str()
		out_dict_list = compute_acuuracy(filename,subject_name_date_list=subject_name_date_list,to_plot=to_plot)
	else:
		for night in subject_name_date_list:
			try:
				subject_label = night["SUBJECT"]
				date_label = night["DATE"]
				print "Computing for",subject_label, "on date", date_label
				accuracy, correlation = compute_accuracy_of_one_night(hdf_obj,subject_label,date_label,to_plot=to_plot)
				out_dict = {}
				out_dict["SUBJECT"]     = subject_label
				out_dict["DATE"]        = date_label
				out_dict["ACCURACY"]    = accuracy
				out_dict["CORRELATION"] = correlation
				out_dict_list.append(out_dict)
			except:
				continue
	return out_dict_list


subject_name_date_list = [{"SUBJECT":"Subject25","DATE":"16112017"}]
output = compute_acuuracy("master_data.hdf",subject_name_date_list,to_plot=1)

print output

out_summary_file = "acc_out3.csv"
if not os.path.isfile(out_summary_file):
	fo = open(out_summary_file,'w')
	fo.close()
fo = open(out_summary_file,'ab')

csv_writer =  csv.writer(fo)
for night in output:
	csv_writer.writerow([night["SUBJECT"],night["DATE"],night["ACCURACY"],night["CORRELATION"]])
