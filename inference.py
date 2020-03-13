from deepsleep import InferenceHeartSequenceLoader
import numpy as np
import os, glob
# from data_generator import InferenceHeartSequenceLoader
from keras.models import load_model, model_from_json


def prepare_data():
    HDF_DIR = os.path.join(os.getcwd(), "deepsleep/data")
    for hdf_file in glob.glob(HDF_DIR + '/' + '*.hdf'):
        print hdf_file
        # hdf_file = HDF_DIR + '/subject36.hdf'
        prep_data = PrepareDataset(hdf_file, seq_len=7500, batch_size=64, n_classes=4)
        prep_data.prep_dataset()


def load_data(subject_file):
    loader = InferenceHeartSequenceLoader(seq_len=7500, batch_size=16, n_classes=4)
    heart_signal, labels = loader.get_data(subject_file)

    return heart_signal, labels


def initialize_model(model_name):
    model = load_model(model_name)

    return model


def initialize_model_json(json_model, model_weights):
    loaded_model = model_from_json(json_model)
    loaded_model = loaded_model.load_weights(model_weights)

    return loaded_model


def inference(model, heart_data):
    predictions = model.predict(heart_data)
    pred_class = predictions.argmax(axis=-1)

    return predictions, pred_class


def evaluate_inference(model, heart_data, labels):
    score = model.evaluate(x=heart_data, y=labels)

    # print score
    return score


if __name__ == '__main__':
    # prepare_data()

    subject_data = "Subject17_03082017" + '.npz'
    model_path = "model_output/8b2fResReg1Val_PTr_128/"
    model_arch = model_path + "model_arch.json"
    model_name = model_path + 'weights.29-0.73-0.92.hdf5'

    model_name = model_path + "full_model.hdf5"

    heart_signal, labels = load_data(subject_data)
    model = initialize_model(model_name)
    # model = initialize_model_json(model_arch, model_name)
    predictions, classes = inference(model, heart_signal)
    # score = evaluate_inference(model, heart_signal, labels)
    print classes
