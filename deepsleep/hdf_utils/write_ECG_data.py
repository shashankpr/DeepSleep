from hdf import HDF
import csv
import datetime as dt
import numpy as np
from astropy.stats import LombScargle

nimhans_time = ["2017-10-20 00:07:23", "2017-10-21 01:27:30", "2017-10-25 22:27:03", "2017-10-27 00:29:34"]
subject = ["Subject18", "Subject19", "Subject21", "Subject21"]
date_vals = ["19102017", "20102017", "25102017", "26102017"]

def set_NIMHANS_data():
    hdf_obj = HDF('tester3.hdf')
    for data_of in range(len(subject)):
        nimhans_start_time = nimhans_time[data_of]
        date_str = date_vals[data_of]
        subject_str = subject[data_of]
        filename = "/home/turtleshelltech/vibhor/dozee1.1/dozee-filter/jj_values/NIMHANS/" + date_str + "_" + subject_str +\
            "_M" + "/heart.csv"
        usefile = open(filename, 'r')
        csv_reader = csv.reader(usefile)
        fmt = '%Y-%m-%d %H:%M:%S'
        time_list = []
        rr_interval_list = []
        for reader in csv_reader:
            time_list.append(dt.datetime.strptime(nimhans_start_time, fmt) + \
                                    dt.timedelta(seconds = float(reader[0])))
            rr_interval_list.append(float(reader[1]))
        filename = "/home/turtleshelltech/vibhor/dozee1.1/dozee-filter/jj_values/NIMHANS/" + date_str + "_" + subject_str + \
                   "_M" + "/stages.csv"
        usefile = open(filename, 'r')
        csv_reader = csv.reader(usefile)
        stages_time_list = []
        stages_list = []
        for reader in csv_reader:
            stages_time_list.append(dt.datetime.strptime(reader[0], fmt))
            stages_list.append(int(reader[1]))

        date_handle = hdf_obj.get_date_handle(subject_str, date_str)
        night_start_time = hdf_obj.get_night_start_time(date_handle = date_handle)

        validation_r_time_list = for_validation_r_time(time_list, night_start_time)

        validation_time_list, heart_rate_list, night_stage_list = for_validation_data(night_start_time=night_start_time,
                                                                                      time_list=time_list,
                                                                                      rr_interval_list=rr_interval_list,
                                                                                      stages_time_list=stages_time_list,
                                                                                      stages_list=stages_list)

        # print night_stage_list
        rmssd_list = calculateRMSSD(time_list, rr_interval_list, night_start_time)
        sdnn_list = calculateSDNN(time_list, rr_interval_list, night_start_time)

        vlfValue, lfValue, hfValue, relativeHF = calculateFreqDomainHRV(time_list, rr_interval_list, night_start_time)
        vlf1_list, vlf2_list, vlf3_list, lf1_list, lf2_list, lf3_list, hf1_list, hf2_list, hf3_list, rel_hf1_list, \
        rel_hf2_list, rel_hf3_list = for_ten_seconds_frequency(vlfValue, lfValue, hfValue, relativeHF)

        # vlf1_list, vlf2_list, vlf3_list, lf1_list, lf2_list, lf3_list, hf1_list, hf2_list, hf3_list, rel_hf1_list, \
        # rel_hf2_list, rel_hf3_list = [], [], [], [], [], [], [], [], [], [], [], []
        for value_change in range(len(heart_rate_list)):
            if np.isnan(heart_rate_list[value_change]):
                print value_change
                heart_rate_list[value_change] = -1
                vlf1_list.insert(value_change, -1)
                vlf2_list.insert(value_change, -1)
                vlf3_list.insert(value_change, -1)
                lf1_list.insert(value_change, -1)
                lf2_list.insert(value_change, -1)
                lf3_list.insert(value_change, -1)
                hf1_list.insert(value_change, -1)
                hf2_list.insert(value_change, -1)
                hf3_list.insert(value_change, -1)
                rel_hf1_list.insert(value_change, -1)
                rel_hf2_list.insert(value_change, -1)
                rel_hf3_list.insert(value_change, -1)

        for add_Value in range(len(heart_rate_list) - len(sdnn_list)):
            sdnn_list.insert(add_Value, -1)
            rmssd_list.insert(add_Value, -1)
        for add_Value in range(len(heart_rate_list) - len(vlf1_list)):
            vlf1_list.insert(add_Value, -1)
            vlf2_list.insert(add_Value, -1)
            vlf3_list.insert(add_Value, -1)
            lf1_list.insert(add_Value, -1)
            lf2_list.insert(add_Value, -1)
            lf3_list.insert(add_Value, -1)
            hf1_list.insert(add_Value, -1)
            hf2_list.insert(add_Value, -1)
            hf3_list.insert(add_Value, -1)
            rel_hf1_list.insert(add_Value, -1)
            rel_hf2_list.insert(add_Value, -1)
            rel_hf3_list.insert(add_Value, -1)

        print len(sdnn_list), len(time_list), len(rmssd_list), len(validation_time_list), len(vlf1_list)

        hdf_obj.set_validation_data(subject_str, date_str, validation_r_time_list=validation_r_time_list,
                                    validation_time_list = validation_time_list, rr_interval_list = rr_interval_list,
                                    heart_rate_list = heart_rate_list, breathing_rate_list = [],
                                    stages_list = night_stage_list, sdnn_list = sdnn_list,	rmssd_list = rmssd_list,
                                    vlf1_list = vlf1_list, vlf2_list = vlf2_list, vlf3_list = vlf3_list,
                                    lf1_list = lf1_list, lf2_list = lf2_list, lf3_list = lf3_list, hf1_list = hf1_list,
                                    hf2_list = hf2_list, hf3_list = hf3_list, rel_hf1_list = rel_hf1_list,
                                    rel_hf2_list = rel_hf2_list, rel_hf3_list = rel_hf3_list, permission_flag = 1)


def set_TLC_data():
    date_str = "19102017"
    subject_str = "Subject18"
    filename = "/home/turtleshelltech/vibhor/dozee1.1/dozee-filter/jj_values/TLC/" + date_str + "_" + subject_str + \
               "_M_TLC" + "/heart.csv"
    usefile = open(filename, 'r')
    csv_reader = csv.reader(usefile)
    fmt = '%Y-%m-%d %H:%M:%S'
    time_list = []
    rr_interval_list = []
    for reader in csv_reader:
        time_list.append(dt.datetime.strptime(reader[0], fmt))
        rr_interval_list.append(float(reader[1]) / 1000.0)

    night_start_time = for_night_start_time(subject_str, date_str)

    validation_r_time_list = for_validation_r_time(time_list, night_start_time)

    validation_time_list, heart_rate_list, night_stage_list = for_validation_data(night_start_time=night_start_time,
                                                                                  time_list=time_list,
                                                                                  rr_interval_list=rr_interval_list)

    rmssd_list = calculateRMSSD(time_list, rr_interval_list)
    sdnn_list = calculateSDNN(time_list, rr_interval_list)
    vlfValue, lfValue, hfValue, relativeHF = calculateFreqDomainHRV(time_list, rr_interval_list)
    vlf1_list, vlf2_list, vlf3_list, lf1_list, lf2_list, lf3_list, hf1_list, hf2_list, hf3_list, rel_hf1_list, \
    rel_hf2_list, rel_hf3_list = for_ten_seconds_frequency(vlfValue, lfValue, hfValue, relativeHF)

    for value_change in range(len(heart_rate_list)):
        if np.isnan(heart_rate_list[value_change]):
            print value_change
            heart_rate_list[value_change] = -1
            vlf1_list.insert(value_change, -1)
            vlf2_list.insert(value_change, -1)
            vlf3_list.insert(value_change, -1)
            lf1_list.insert(value_change, -1)
            lf2_list.insert(value_change, -1)
            lf3_list.insert(value_change, -1)
            hf1_list.insert(value_change, -1)
            hf2_list.insert(value_change, -1)
            hf3_list.insert(value_change, -1)
            rel_hf1_list.insert(value_change, -1)
            rel_hf2_list.insert(value_change, -1)
            rel_hf3_list.insert(value_change, -1)

    for add_Value in range(len(heart_rate_list) - len(sdnn_list)):
        sdnn_list.insert(add_Value, -1)
        rmssd_list.insert(add_Value, -1)
    for add_Value in range(len(heart_rate_list) - len(vlf1_list)):
        vlf1_list.insert(add_Value, -1)
        vlf2_list.insert(add_Value, -1)
        vlf3_list.insert(add_Value, -1)
        lf1_list.insert(add_Value, -1)
        lf2_list.insert(add_Value, -1)
        lf3_list.insert(add_Value, -1)
        hf1_list.insert(add_Value, -1)
        hf2_list.insert(add_Value, -1)
        hf3_list.insert(add_Value, -1)
        rel_hf1_list.insert(add_Value, -1)
        rel_hf2_list.insert(add_Value, -1)
        rel_hf3_list.insert(add_Value, -1)

    hdf_obj.set_validation_data(subject_str, date_str, validation_r_time_list=validation_r_time_list,
                                validation_time_list=validation_time_list, rr_interval_list=rr_interval_list,
                                heart_rate_list=heart_rate_list, breathing_rate_list=[], stages_list=night_stage_list,
                                sdnn_list=sdnn_list, rmssd_list=rmssd_list, vlf1_list=vlf1_list,
                                vlf2_list=vlf2_list, vlf3_list=vlf3_list, lf1_list=lf1_list, lf2_list=lf2_list,
                                lf3_list=lf3_list, hf1_list=hf1_list, hf2_list=hf2_list, hf3_list=hf3_list,
                                rel_hf1_list=rel_hf1_list, rel_hf2_list=rel_hf2_list, rel_hf3_list=rel_hf3_list,
                                permission_flag=1)


def calculateRMSSD(timeList, rrDistance, night_start_time):
    startTime = night_start_time
    rrDistanceDiff = []
    rmssd = []
    rmssdTime = []
    print startTime
    for timeVal in range(len(timeList) - 1):
        if timeList[timeVal] - startTime < dt.timedelta(seconds=30):
            rrDistanceDiff.append((rrDistance[timeVal + 1] - rrDistance[timeVal]) ** 2)
        else:
            rmssd.append((np.mean(rrDistanceDiff) ** 0.5) * 1000)
            if np.isnan(rmssd[-1]):
                rmssd[-1] = -1
            rmssdTime.append(startTime)
            startTime = startTime + dt.timedelta(seconds=30)
            rrDistanceDiff = []
            rrDistanceDiff.append((rrDistance[timeVal + 1] - rrDistance[timeVal]) ** 2)
    return rmssd

def calculateSDNN(timeList, rrDistance, night_start_time):
    sdnnTime = []
    sdnnValue = []
    rrDiff = []
    startTime = night_start_time
    for timeVal in range(len(timeList) - 1):
        if timeList[timeVal] - startTime < dt.timedelta(seconds=30):
            rrDiff.append(rrDistance[timeVal])
        else:
            sdnnValue.append(np.std(rrDiff) * 1000)
            if np.isnan(sdnnValue[-1]):
                sdnnValue[-1] = -1
            sdnnTime.append(startTime)
            startTime = startTime + dt.timedelta(seconds=30)
            rrDiff = []
            rrDiff.append(rrDistance[timeVal])
    return sdnnValue

def calculateFreqDomainHRV(timeList, rrDistance, night_start_time):
    fmt = '%Y-%m-%d %H:%M:%S'
    startTime = night_start_time
    # print startTime
    jjDiff = []
    vlfValue = []
    vlfTime = []
    lfValue = []
    lfTime = []
    hfValue = []
    hfTime = []
    relativeHF = []
    varianceHF = []
    jjDisTimeC = []
    check = 0
    freq = np.linspace(0.0001, 0.5, 100000)
    vlfBand = (0, 0.04)
    lfBand = (0.04, 0.15)
    hfBand = (0.15, 0.4)
    for j in range(len(timeList)):
        if j != 0 and timeList[-1] - startTime >= dt.timedelta(0, 300) and check == 0:
            startTime = startTime + dt.timedelta(seconds=10)
        if timeList[j] < startTime:
            check = 1
            continue
        else:
            check = 0
        for i in range(j, len(timeList) - 1):
            if timeList[i] - startTime <= dt.timedelta(0, 300):
                if abs(rrDistance[i + 1] - rrDistance[i]) <= 0.5:
                    jjDiff.append(np.float64(rrDistance[i]))
                    jjDisTimeC.append(dt.timedelta.total_seconds(timeList[i] - startTime))
            else:
                if len(jjDiff) >= 2:
                    jjDiffArray = np.array(jjDiff)
                    jjDisTimeCArray = np.array(jjDisTimeC)

                    power = LombScargle(jjDisTimeCArray, jjDiffArray).power(freq)

                    vlf, lf, hf, total_power, relativeHFValue, varianceHFValue = _auc(freq, power, vlfBand, lfBand, hfBand)
                    vlfValue.append(vlf * 1000)
                    vlfTime.append(dt.datetime.strftime(startTime, fmt))
                    lfValue.append(lf * 1000)
                    lfTime.append(dt.datetime.strftime(startTime, fmt))
                    hfValue.append(hf * 1000)
                    hfTime.append(dt.datetime.strftime(startTime, fmt))
                    relativeHF.append(relativeHFValue * 1000)
                    varianceHF.append(varianceHFValue * 1000)
                else:
                    vlfValue.append(-1)
                    vlfTime.append(-1)
                    lfValue.append(-1)
                    lfTime.append(-1)
                    hfValue.append(-1)
                    hfTime.append(-1)
                    relativeHF.append(-1)
                    varianceHF.append(-1)

                jjDiff = []
                jjDisTimeC = []
                break
        if j % 500 == 0:
            print j
    return vlfValue, lfValue, hfValue, relativeHF


def _auc(fxx, pxx, vlf_band, lf_band, hf_band):
    vlf_indexes = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
    lf_indexes = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
    hf_indexes = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])
    vlf = np.trapz(y=pxx[vlf_indexes], x=fxx[vlf_indexes])
    lf = np.trapz(y=pxx[lf_indexes], x=fxx[lf_indexes])
    hf = np.trapz(y=pxx[hf_indexes], x=fxx[hf_indexes])
    relative_HF = max(pxx[hf_indexes]) / sum(pxx[hf_indexes])
    variance_HF = np.var(pxx[hf_indexes])
    total_power = vlf + lf + hf

    return vlf, lf, hf, total_power, relative_HF, variance_HF


def for_night_start_time(subject_str, date_str):
    date_handle = hdf_obj.get_date_handle(subject_str, date_str)
    night_start_time = hdf_obj.get_night_start_time(date_handle=date_handle)
    return night_start_time

def for_validation_r_time(time_list, night_start_time):
    validation_r_time_list = []
    for time_value in time_list:
        validation_r_time_list.append((time_value - night_start_time).total_seconds())
    return validation_r_time_list

def for_validation_data(night_start_time=None, time_list=[], rr_interval_list=[], stages_time_list=[], stages_list=[]):
    validation_time_list = []
    heart_rate_list = []
    rpeak_value = []
    night_time = night_start_time
    night_stage_list = []
    for time_value in range(len(time_list)):
        if night_start_time - time_list[time_value] <= dt.timedelta(seconds=14) and \
                                night_start_time - time_list[time_value] >= dt.timedelta(seconds=0) or \
                                        time_list[time_value] - night_start_time >= dt.timedelta(seconds=0) and \
                                        time_list[time_value] - night_start_time <= dt.timedelta(seconds=15):
            rpeak_value.append(60.0 / rr_interval_list[time_value])
        elif time_list[time_value] - night_start_time > dt.timedelta(seconds=15):
            if stages_time_list != []:
                for stages_time in range(len(stages_time_list) - 1):
                    if stages_time == 0 and night_start_time <= stages_time_list[stages_time]:
                        night_stage_list.append(stages_list[stages_time])
                        break
                    elif night_start_time >= stages_time_list[stages_time] and night_start_time < \
                            stages_time_list[stages_time + 1]:
                        night_stage_list.append(stages_list[stages_time])
                        break
                    elif night_start_time >= stages_time_list[-1]:
                        night_stage_list.append(stages_list[-1])
                        break

            validation_time_list.append((night_start_time - night_time).total_seconds())
            heart_rate_list.append(np.mean(rpeak_value))
            rpeak_value = []
            rpeak_value.append(60.0 / rr_interval_list[time_value])
            night_start_time = night_start_time + dt.timedelta(seconds=30)
    return validation_time_list, heart_rate_list, night_stage_list

def for_ten_seconds_frequency(vlfValue, lfValue, hfValue, relativeHF):
    vlf1_list = []
    vlf2_list = []
    vlf3_list = []
    lf1_list = []
    lf2_list = []
    lf3_list = []
    hf1_list = []
    hf2_list = []
    hf3_list = []
    rel_hf1_list = []
    rel_hf2_list = []
    rel_hf3_list = []

    for data_points in range(9):
        vlf1_list.append(-1)
        vlf2_list.append(-1)
        vlf3_list.append(-1)
        lf1_list.append(-1)
        lf2_list.append(-1)
        lf3_list.append(-1)
        hf1_list.append(-1)
        hf2_list.append(-1)
        hf3_list.append(-1)
        rel_hf1_list.append(-1)
        rel_hf2_list.append(-1)
        rel_hf3_list.append(-1)

    for element in range(0, len(relativeHF) - 3, 3):
        vlf1_list.append(vlfValue[element])
        vlf2_list.append(vlfValue[element + 1])
        vlf3_list.append(vlfValue[element + 2])

        lf1_list.append(lfValue[element])
        lf2_list.append(lfValue[element + 1])
        lf3_list.append(lfValue[element + 2])

        hf1_list.append(hfValue[element])
        hf2_list.append(hfValue[element + 1])
        hf3_list.append(hfValue[element + 2])

        rel_hf1_list.append(relativeHF[element])
        rel_hf2_list.append(relativeHF[element + 1])
        rel_hf3_list.append(relativeHF[element + 2])

    return vlf1_list, vlf2_list, vlf3_list, lf1_list, lf2_list, lf3_list, hf1_list, hf2_list, hf3_list, rel_hf1_list, \
           rel_hf2_list, rel_hf3_list

hdf_obj = HDF('tester3.hdf')
set_NIMHANS_data()