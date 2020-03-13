from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping, TensorBoard
from keras.models import save_model, load_model

import numpy as np
import time
import logging
import os
import json, io

import model_callbacks
from data_generator import HeartSequenceLoader, HeartSequenceGenerator
from data_loader import DataLoader
from models import BuildGPUModel


class Trainer(BuildGPUModel):
    def __init__(self, gpu_model,
                 model,
                 n_epochs,
                 batch_size,
                 training_files,
                 validation_files=None,
                 test_files=None,
                 shuffle_file_order=False,
                 nb_workers=2,
                 model_name='trained_model',
                 class_weight=None,
                 is_pretrain=False,
                 val_pretrain=False,
                 is_stateful_train=False):


        self.logger = logging.getLogger(__name__)
        self.LOG_DIR = os.path.join(os.getcwd(), "deepsleep/logs/")
        self.MODEL_DIR = os.path.join(os.getcwd(), 'model_output/')

        self.gpu_model = gpu_model
        self.model = model
        self.n_epochs = n_epochs
        self.training_files = training_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.shuffle_files = shuffle_file_order
        self.nb_workers = nb_workers
        self.class_weight = class_weight
        self.is_pretrain = is_pretrain
        self.val_pretrain = val_pretrain
        self.is_stateful_train = is_stateful_train

        super(Trainer, self).__init__(BATCH_SIZE=self.batch_size)
        self.seq_len = self.SEQ_LEN
        self.n_classes = self.n_classes

        name = model_name
        if self.is_pretrain:
            name = model_name + '_PTr_' + str(self.batch_size)
        if self.is_stateful_train:
            name = model_name + '_SqTr_' + str(self.batch_size)

        self.model_name = self.log_name = name
        model_folder = os.path.join(self.MODEL_DIR, self.model_name)

        try:
            os.mkdir(model_folder)
        except:
            os.rmdir(model_folder)
            os.mkdir(model_folder)
        self.save_path = model_folder

    def train_model(self):
        """

        Returns:

        """
        start = time.time()

        # Initialize models

        training_model = None  # Model to use for training
        saving_model = None  # Model to use for saving weights

        # Check for GPU-optimized model
        if self.gpu_model:
            training_model = self.gpu_model
            saving_model = self.model

        if self.gpu_model is None:
            training_model = self.model
            saving_model = self.model

        # Save the original model and not the GPU model
        # Save model architecture
        json_arch = saving_model.to_json()

        with io.open(os.path.join(self.save_path, 'model_arch.json'), 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(json_arch, sort_keys=True, indent=4, ensure_ascii=False)))

        # Prepare the callbacks
        callbacks_list = self.prepare_callbacks()

        self.logger.info("Loading Generators ...")
        training_generator = HeartSequenceGenerator(seq_len=self.seq_len,
                                                    batch_size=self.batch_size,
                                                    n_classes=self.n_classes,
                                                    is_pretrain=self.is_pretrain,
                                                    is_stateful_train=self.is_stateful_train
                                                    )

        self.logger.debug("Getting Validation files = {}".format(self.validation_files))
        validation_data = HeartSequenceGenerator(   seq_len=self.seq_len,
                                                    batch_size=self.batch_size,
                                                    n_classes=self.n_classes,
                                                    is_pretrain=self.val_pretrain,
                                                    is_stateful_train=self.is_stateful_train
                                                    )

        val_heart, val_labels = validation_data.validation_sequence(self.validation_files)

        # dl = DataLoader()
        # dl.get_validation_report(val_heart, val_labels)

        self.logger.info("Training & Validating on the Data...")

        training_model.fit_generator(generator=training_generator.generate_sequences(self.training_files),
                                     steps_per_epoch=training_generator.total_batch_count(self.training_files),
                                     validation_data=(val_heart, val_labels),
                                     # validation_steps=validation_generator.get_batch_count(self.validation_files),
                                     epochs=self.n_epochs,
                                     callbacks=callbacks_list,
                                     verbose=1,
                                     use_multiprocessing=False,
                                     workers=self.nb_workers,
                                     max_queue_size=40,
                                     shuffle=False,
                                     )
        score, acc = training_model.evaluate(x=val_heart, y=val_labels, batch_size=self.batch_size)

        print "Evaluation Score = {}".format(score)
        print "evaluation Acc = {}".format(acc)

        self.logger.info("> Compilation Time : {}".format(time.time() - start))

        # Save full model
        saving_model.save(os.path.join(self.save_path, 'full_model.hdf5'))


    def prepare_callbacks(self):

        shuffle_training_files = LambdaCallback(on_epoch_end=lambda epoch, logs: np.random.shuffle(self.training_files))

        history = model_callbacks.Histories()
        csv_logger = CSVLogger(filename=os.path.join(self.save_path,self.log_name), append=False)

        checkpoint_file = os.path.join(self.save_path, 'weights.{epoch:02d}-{acc:.2f}-{val_loss:.2f}.hdf5')
        checkpointer = ModelCheckpoint(filepath=checkpoint_file,
                                       monitor='val_loss',
                                       mode='min',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

        # Reduce learning rate on reaching a plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=15, min_lr=0.5e-6, verbose=1)

        # Create EarlyStopping if no improvement in the validation loss
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=int(30), verbose=1, mode='auto')

        # Tensorboard
        tensorb = TensorBoard(log_dir=self.save_path, histogram_freq=5, write_grads=True, write_images=True, batch_size= self.batch_size)

        # Print the batch number at the beginning of every batch.
        batch_print_callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.logger.debug(("Batch No: {}, Batch size: {}".format(batch, logs.get('size')))))

        state_resetter = model_callbacks.ResetModel(self.training_files, self.batch_size)

        default_list = [history,
                        csv_logger,
                        checkpointer,
                        reduce_lr,
                        early_stop,
                        # tensorb,
                        # batch_print_callback,
                        ]

        if self.shuffle_files and self.is_stateful_train:
            callbacks_list = default_list + [shuffle_training_files, state_resetter]

        elif self.is_stateful_train:
            callbacks_list = default_list + [state_resetter]

        elif self.shuffle_files:
            callbacks_list = default_list + [shuffle_training_files]

        else:
            callbacks_list = default_list

        return callbacks_list