from hdf import HDF

def read_processed_data():
    hdf_obj = HDF('tester2.hdf')
    processed_main_dictionary = hdf_obj.get_processed_data("Subject18", "19102017", start_time = "2017-10-20 02:00:00",
                                                           end_time = "2017-10-20 03:00:00",
                                                           column_name_values_for_snoring = ["Snoring Time", "Duration"],
                                                           column_name_values_for_processed = [])
    print processed_main_dictionary['JJ_TABLE']
    # print processed_main_dictionary['PROCESSED_TABLE']
    # print processed_main_dictionary['SNORING_TABLE']
    # print processed_main_dictionary['ZERO_CROSSINGS']


def read_raw_data():
    hdf_obj = HDF('tester2.hdf')
    raw_dictionary = hdf_obj.get_raw_data("Subject18", "19102017", start_time = "2017-10-20 02:00:00",
                                          end_time = "2017-10-20 03:00:00")
    print raw_dictionary.keys()

def read_filtered_data():
    hdf_obj = HDF('tester2.hdf')
    filtered_dictionary = hdf_obj.get_filtered_data("Subject18", "19102017", start_time = "2017-10-20 02:00:00",
                                                    end_time = "2017-10-20 03:00:00")
    print filtered_dictionary.keys()

def read_validation_data():
    hdf_obj = HDF('tester2.hdf')
    validation_dictionary = hdf_obj.get_validation_data("Subject18", "19102017", column_name_values_for_validation = [],
							start_time = "2017-10-20 02:00:00", end_time = "2017-10-20 03:00:00")
    validation_dictionary.keys()

read_validation_data()