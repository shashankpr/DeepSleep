
from hdf import HDF

filename = "test.hdf"
hdf_object = HDF(filename)

tlist   =  [10,20,30,40,50,60,70,80,90,100]
hr_list =  [60,61,62,63,64,65,66,67,68,69]
br_list =  [20,21,22,23,24,25,26,27,28,29]
m_list  =  [1,2,3,4,5,6,7,8,9,10]

# hdf_object.append_processed_data("GAURAV", "12121991", time_list = tlist, heart_rate_list = hr_list, breathing_rate_list = br_list, call_number=0)
#hdf_object.append_raw_data()
#hdf_object.append_filtered_data()
#hdf_object.append_validation_data()
#, breathing_rate_list = br_list

# hdf_object.set_raw_data("GAURAV", "12121991", epoch_number = 1, raw_time_list = tlist, piezo_value_list = hr_list)

# hdf_object.set_filtered_data("GAURAV", "12121991", epoch_number = 1, filtered_time_list = tlist, heart_value_list = hr_list,
# 						  breathing_value_list = br_list, movement_value_list = m_list)

# hdf_object.append_validation_data("GAURAV", "12121991",  validation_time_list = tlist, heart_rate_list = hr_list, stages_list = m_list)

hdf_object.get_processed_data("GAURAV", "12121991")