from deepsleep import DataLoader, Trainer, BuildGPUModel, PrepareDataset, PreparePretrainDataset
import tensorflow as tf
import os, glob

import argparse

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rt", "--reptrain", help="Compile and run REPR Model", action="store_true")
    parser.add_argument("-sq", "--seqtrain", help="Compile and run SEQ Model", action="store_true")
    parser.add_argument("-ft", "--finetune", help="Compile and run Finetune network", action="store_true")

    parser.add_argument("-t", "--test", help="Run tests", action="store_true")
    parser.add_argument("-tr", "--train", help="Train models", action="store_true")
    parser.add_argument("-p", "--prepare", help="Prepare and extract dataset from HDF", action="store_true")
    parser.add_argument("-pp", "--preparePretrain", help="Prepare and extract dataset from HDF", action="store_true")
    parser.add_argument("-r", "--report", help="Get dataset report", action="store_true")

    args = parser.parse_args()

    if args.preparePretrain:
        HDF_DIR = os.path.join(os.getcwd(), "deepsleep/data")
        # hdf_file = HDF_DIR + '/Subject8.hdf'
        prep_data = PreparePretrainDataset(sampling_mode='random_over_sample', seq_len=7500, batch_size=64, n_classes=4)
        prep_data.store_pretrain_files()
        # HDF_DIR = os.path.join(os.getcwd(), "deepsleep/data")
        # for hdf_file in glob.glob(HDF_DIR + '/' + '*.hdf'):
        #     print hdf_file
        #     prep_data = PrepareNonEpochDataset(hdf_file)
        #     prep_data.store_dataset_to_npy(sampling_mode='over_sample')

    if args.prepare:
        HDF_DIR = os.path.join(os.getcwd(), "deepsleep/data")
        # hdf_file = HDF_DIR + '/Subject41.hdf'
        # prep_data = PrepareDataset(hdf_file, seq_len=7500, batch_size=64, n_classes=4)
        # prep_data.prep_dataset()
        HDF_DIR = os.path.join(os.getcwd(), "deepsleep/data")
        for hdf_file in glob.glob(HDF_DIR + '/' + '*.hdf'):
            print hdf_file
            # hdf_file = HDF_DIR + '/subject36.hdf'
            prep_data = PrepareDataset(hdf_file, seq_len=7500, batch_size=64, n_classes=4)
            prep_data.prep_dataset()

    if args.report:
        NPZ_DIR = os.path.join(os.getcwd(), 'dataset')
        npz_file = NPZ_DIR + '/' + 'Subject30_16042018.npz'
        prep_nonepoch = PrepareNonEpochDataset(npz_file)
        prep_nonepoch.get_dataset_report(data=npz_file)

    if args.train:
        if args.seqtrain:
            load_data = DataLoader(split_criterion=0.1, is_pretrain=False)
            training_files, validation_files = load_data.split_dataset(val_count=1)

            print "Running GPU-optimized models"
            build_gpu_model = BuildGPUModel(BATCH_SIZE=64, SAMPLE_RATE=250)

            model = build_gpu_model.deepsleep_network(cnn_residual_connection=False,
                          finetune=False,
                          pretrain_model_name='pretrain_weights.h5',
                          blocks=4,
                          num_layers=2,
                          filter_block=2,
                          maxpool_block=2,
                          learning_rate=1e-4,
                          num_filters=64,
                          kernel_size=16,
                          kernel_regularizer=None,
                          dropout_rate=None,
                          plot_arch=False)


            trainer = Trainer(gpu_model=None,
                              model=model,
                              training_files=training_files,
                              validation_files=validation_files,
                              n_epochs=200,
                              batch_size=64,
                              nb_workers=1,
                              shuffle_file_order=True,
                              model_name='seqtrain_2LS_tr',
                              class_weight=None,
                              is_pretrain=False,
                              val_pretrain=False,
                              is_stateful_train=True)
            trainer.train_model()

        if args.reptrain:
            load_data = DataLoader(split_criterion=0.1, is_pretrain=False)
            training_files, validation_files = load_data.split_dataset(val_count=5)

            print "Running GPU-optimized models"
            build_gpu_model = BuildGPUModel(BATCH_SIZE=128, SAMPLE_RATE=250, n_gpus=2)

            pretrain_model = build_gpu_model.representation_layer(blocks=10,
                                                       num_layers=2,
                                                       filter_block=4,
                                                       maxpool_block=2,
                                                       learning_rate=1e-3,
                                                       num_filters=64,
                                                       kernel_size=16,
                                                       dropout_rate=0.3,
                                                       kernel_regularizer=None,
                                                       plot_arch=False,
                                                       residual_connection=True,
                                                       include_top=True)

            trainer = Trainer(gpu_model=pretrain_model[0],
                              model=pretrain_model[1],
                              training_files=training_files,
                              validation_files=validation_files,
                              n_epochs=100,
                              batch_size=128,
                              nb_workers=1,
                              shuffle_file_order=True,
                              model_name='10b2lRes5Val',
                              class_weight=None,
                              is_pretrain=False,
                              val_pretrain=True)
            trainer.train_model()

        if args.finetune:
            load_data = DataLoader(split_criterion=0.1, is_pretrain=False)
            training_files, validation_files = load_data.split_dataset(val_count=1)

            print "Running GPU-optimized models"
            build_gpu_model = BuildGPUModel(BATCH_SIZE=128, SAMPLE_RATE=250)
            model = build_gpu_model.deepsleep_network(cnn_residual_connection=True,
                          finetune=True,
                          pretrain_model_name='8b2lRes_PTr_128',
                          blocks=8,
                          num_layers=2,
                          filter_block=2,
                          maxpool_block=2,
                          learning_rate=1e-5,
                          num_filters=64,
                          kernel_size=16,
                          kernel_regularizer=None,
                          dropout_rate=0.3,
                          is_stateful=False,
                          plot_arch=False)

            trainer = Trainer(gpu_model=model[0],
                              model=model[1],
                              training_files=training_files,
                              validation_files=validation_files,
                              n_epochs=100,
                              batch_size=128,
                              nb_workers=1,
                              shuffle_file_order=True,
                              model_name='finetune_model',
                              class_weight=None,
                              is_pretrain=False,
                              is_stateful_train=False)
            trainer.train_model()

        if args.test:
            load_data = DataLoader(split_criterion=0.1, is_pretrain=False)
            training_files, validation_files = load_data.split_dataset(val_count=1)

            print "Running GPU-optimized models"
            build_gpu_model = BuildGPUModel(BATCH_SIZE=64, SAMPLE_RATE=250)
            model = build_gpu_model.deepsleep_network()

            trainer = Trainer(gpu_model=None,
                              model=model,
                              training_files=training_files,
                              validation_files=validation_files,
                              n_epochs=1,
                              batch_size=64,
                              nb_workers=1,
                              shuffle_file_order=True,
                              model_name='seqtrain_test',
                              class_weight=None,
                              is_pretrain=False)
            trainer.train_model()


def run_test():
    # load_data = DataLoader(split_criterion=0.2)
    # training_files, validation_files = load_data.split_dataset()
    # load_data.get_dataset_report(training_data=training_files, validation_data=validation_files)
    # # load_data.balance_class(training_data=training_files)

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Train models", action="store_true")
    parser.add_argument("-p", "--prepare", help="Prepare and extract dataset from HDF", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        HDF_DIR = os.path.join(os.getcwd(), "tests/test_dataset")
        hdf_file = HDF_DIR + '/subject17.hdf'
        print hdf_file


    if args.train:
        from tests import train

        #train.test_training()
        train.test_training_2()

if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #print sess
    run()
