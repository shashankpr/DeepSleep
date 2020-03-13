from keras import losses, metrics, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate, add, Reshape
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM, Masking, Flatten, Bidirectional, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.utils import print_summary, multi_gpu_model, plot_model

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

import logging
import os


# seed = 7
# np.random.seed(seed=seed)

class BuildGPUModel(object):
    def __init__(self, BATCH_SIZE, SAMPLE_RATE=250, n_classes=4, FEATURES_DIM=1, n_gpus=2):
        """Initializing the hyper-parameters

        Args:
            file_path        : Path to the HDF file
            BATCH_SIZE  (int): Number of batches of data to be supplied to the model_output
            SEQ_LEN     (int): Number of sequences to be visited/remembered before processing the next sample point.
            n_classes   (int): Number of classes/outputs of the training data.
            FEATURES_DIM (int): Total number of features to consider for training. 1 = Heart, 7 = HRV features
        """

        # self.file_path = file_path
        self.SEQ_LEN = SAMPLE_RATE * 30
        self.BATCH_SIZE = BATCH_SIZE
        self.FEATURES_DIM = FEATURES_DIM
        self.n_classes = n_classes
        self.n_gpus = n_gpus
        self.sr = SAMPLE_RATE
        self.MODEL_DIR = os.path.join(os.getcwd(), 'model_output/')
        self.logger = logging.getLogger(__name__)

    def test_model(self):
        """

        Returns:

        """
        layers = [self.FEATURES_DIM, 10, self.n_classes]
        model = Sequential()

        model.add(Masking(mask_value=0., input_shape=(self.SEQ_LEN, self.FEATURES_DIM)))
        model.add(LSTM(
            units=layers[1],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            units=self.n_classes,
            activation='softmax'))

        gpu_model = multi_gpu_model(model, gpus=self.n_gpus)
        gpu_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        print_summary(gpu_model, print_fn=self.logger.info)

        return (gpu_model, model)

    def simple_cnn_model(self):

        model = Sequential()
        # model.add(Masking(mask_value=0., input_shape=(self.SEQ_LEN, self.FEATURES_DIM)))
        model.add(Conv1D(input_shape=(self.SEQ_LEN, self.FEATURES_DIM),
                         filters=32,
                         kernel_size=14,
                         padding='same',
                         name='conv1'))

        model.add(Activation(activation='relu', name='act1'))
        model.add(MaxPooling1D(pool_size=3))

        model.add(GlobalMaxPooling1D())

        model.add(Dense(1024))
        # model.add(BatchNormalization())
        model.add(Activation(activation='relu'))

        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))

        optimizer = Adam(lr=0.002)

        gpu_model = multi_gpu_model(model, gpus=self.n_gpus)
        gpu_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy'])
        return (gpu_model, model)

    def add_conv_layer(self, inputs,
                       num_filters=16,
                       kernel_size=3,
                       strides=1,
                       activation='relu',
                       kernel_regularizer=None,
                       batch_normalization=True,
                       conv_first=True,
                       name='conv'):
        """
        1D Convolution-Batch Normalization-Activation stack builder
        Args:
            inputs: input tensor from input image or previous layer
            num_filters:            Conv1D number of filters
            kernel_size:            Conv1D kernel size
            strides:                Conv1D square stride dimensions
            activation:             activation name
            kernel_regularizer:     kernel_regularizer name
            batch_normalization:    whether to include batch normalization
            conv_first:             conv-bn-activation (True) or
                                    bn-activation-conv (False)
            name:                   Name for the layer

        Returns:
            x (tensor):             tensor as input to the next layer
        """

        if kernel_regularizer is None:
            conv = Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_regularizer=None,
                          name=name)
        else:
            conv = Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_regularizer=regularizers.l2(kernel_regularizer),
                          name=name)

        x = inputs
        x = BatchNormalization()(x)
        if conv_first:
            x = conv(x)
            # if batch_normalization:
            #     x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)

        return x

    def representation_layer(self,
                             input=None,
                             blocks=4,
                             num_layers=2,
                             filter_block=2,
                             maxpool_block=2,
                             learning_rate=1e-2,
                             num_filters=64,
                             kernel_size=16,
                             kernel_regularizer=None,
                             dropout_rate=None,
                             residual_connection=False,
                             include_top=False,
                             plot_arch=False):
        """

        Args:
            input:
            blocks:
            num_layers:
            filter_block:
            maxpool_block:
            num_filters:
            kernel_size:
            kernel_regularizer:
            dropout_rate:
            residual_connection:
            include_top:
            plot_arch:

        Returns:

        """
        print "========== SUMMARY =========="

        print "Sampling Rate = {}".format(self.sr)
        print "Batch Size = {}".format(self.BATCH_SIZE)

        print "{} Blocks of CNN".format(blocks)
        print "Total CNN layers = {}".format(blocks * num_layers)
        print "Maxpool = 2; at the end of {} block".format(maxpool_block)
        print "Droput = {}; at the end of {} block".format(dropout_rate, maxpool_block)
        print "Filter updates after {} blocks".format(filter_block)
        print "repr_{}B{}L_tr_resid{}".format(blocks, num_layers, residual_connection)

        if input is None:
            # Common BCG Input to both the CNN layers
            input_signal = Input(shape=(self.SEQ_LEN, self.FEATURES_DIM), name='input_signal')
        else:
            input_signal = input

        # First CNN layer
        cnn = self.add_conv_layer(inputs=input_signal,
                                  num_filters=num_filters,
                                  kernel_size=100,
                                  strides=4,
                                  activation='relu',
                                  kernel_regularizer=kernel_regularizer,
                                  batch_normalization=True,
                                  conv_first=True,
                                  name='conv')

        x = MaxPooling1D(pool_size=2, name='max')(cnn)
        if dropout_rate is not None:
            x = Dropout(dropout_rate, name='dropout')(x)

        # Intitialize residual connection
        res_input = x

        for block_no in range(1, blocks + 1):
            for layers in range(num_layers):
                layer_name = str(block_no) + str(layers + 1)

                if layers == num_layers - 1:
                    x = self.add_conv_layer(inputs=x,
                                            num_filters=num_filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            activation=None,
                                            kernel_regularizer=kernel_regularizer,
                                            batch_normalization=True,
                                            conv_first=True,
                                            name='conv' + layer_name)
                else:
                    x = self.add_conv_layer(inputs=x,
                                            num_filters=num_filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            activation='relu',
                                            kernel_regularizer=kernel_regularizer,
                                            batch_normalization=True,
                                            conv_first=True,
                                            name='conv' + layer_name)
            if residual_connection:
                x = add([x, res_input])
                x = Activation('relu')(x)

                if dropout_rate is not None:
                    x = Dropout(dropout_rate)(x)

                if block_no % filter_block == 0 and block_no % maxpool_block == 0:
                    num_filters = num_filters * 2
                    res_input = self.add_conv_layer(inputs=x,
                                                    num_filters=num_filters,
                                                    kernel_size=kernel_size,
                                                    strides=1,
                                                    activation=None,
                                                    batch_normalization=False,
                                                    name='res_conv' + str(block_no))
                    res_input = BatchNormalization()(res_input)

                    res_input = MaxPooling1D(pool_size=2, name='res_max' + str(block_no))(res_input)
                    x = MaxPooling1D(pool_size=2, name='max' + str(block_no))(x)

                elif block_no % maxpool_block == 0:
                    res_input = MaxPooling1D(pool_size=2, name='res_max' + str(block_no))(x)
                    x = MaxPooling1D(pool_size=2, name='max' + str(block_no))(x)
                else:
                    res_input = x
            else:
                x = Activation('relu')(x)
                if dropout_rate is not None:
                    x = Dropout(dropout_rate)(x)

                if block_no % filter_block == 0:
                    num_filters = num_filters * 2

                if block_no % maxpool_block == 0:
                    x = MaxPooling1D(pool_size=2, name='max' + str(block_no))(x)

        # Representation layer with the bottleneck features only
        repr_layer = x

        if include_top:
            # Build the representation Learning network
            cnn_top_layer = GlobalAveragePooling1D()(x)
            cnn_top_layer = Dense(self.n_classes, activation='softmax')(cnn_top_layer)

            # Initialize representation/pre-train model
            repr_model = Model(inputs=input_signal, outputs=cnn_top_layer)
            print repr_model.summary()

            if plot_arch:
                if residual_connection:
                    plot_model(repr_model, show_shapes=True, to_file='repr_layer_resid.png')
                else:
                    plot_model(repr_model, show_shapes=True, to_file='repr_layer.png')

            optimizer = Adam(lr=learning_rate)
            try:
                gpu_repr_model = multi_gpu_model(model=repr_model, gpus=self.n_gpus, cpu_relocation=True)
                gpu_repr_model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy,
                                       metrics=['accuracy'])

                return (gpu_repr_model, repr_model)
            except:
                repr_model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])
                return (None, repr_model)

        else:
            return repr_layer

    def sequential_layer(self, input, is_stateful=False):

        # Add masking layer to make the LSTM ignore the padded 0s in the signal
        # Since stateful=True, input_shape = (#batchsize, #seq_len, #dim)

        # with tf.device('/gpu:0'):

        cnn_layer_shape = input.get_shape()
        cnn_dim = int(cnn_layer_shape[-1].value)

        # Bi-directional LSTM - 1
        seq_layer = Masking(mask_value=0.)(input)
        seq_layer = Bidirectional(LSTM(cnn_dim / 2,
                                       return_sequences=True,
                                       stateful=is_stateful,
                                       return_state=True,
                                       name='bi_LSTM_1'), merge_mode='concat')(seq_layer)
        seq_layer = Activation(activation='relu')(seq_layer)
        # seq_layer = Dropout(0.3)(seq_layer)

        # Bi-directional LSTM - 2
        seq_layer = Bidirectional(LSTM(cnn_dim / 2, stateful=is_stateful, name='bi_LSTM_2'), merge_mode='concat')(seq_layer)
        seq_layer = Activation(activation='relu')(seq_layer)
        # seq_layer = Dropout(0.3)(seq_layer)

        return seq_layer

    def deepsleep_network(self, cnn_residual_connection=False,
                          finetune=False,
                          is_stateful=False,
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
                          plot_arch=False):

        # Since stateful=True, input_shape = (#batchsize, #seq_len, #dim)
        if is_stateful:
            input_signal = Input(batch_shape=(self.BATCH_SIZE, self.SEQ_LEN, self.FEATURES_DIM), name='sequential_input')
        else:
            input_signal = Input(shape=(self.SEQ_LEN, self.FEATURES_DIM),
                                 name='sequential_input')

        cnn_repr_layer = self.representation_layer(input=input_signal,
                                                   blocks=blocks,
                                                   num_layers=num_layers,
                                                   num_filters=num_filters,
                                                   filter_block=filter_block,
                                                   maxpool_block=maxpool_block,
                                                   kernel_size=kernel_size,
                                                   kernel_regularizer=kernel_regularizer,
                                                   dropout_rate=dropout_rate,
                                                   residual_connection=cnn_residual_connection,
                                                   include_top=False,
                                                   plot_arch=False)

        cnn_shape = cnn_repr_layer.get_shape()
        cnn_dim = int(cnn_shape[-1].value)

        # Make residual connection
        residual_connection = GlobalAveragePooling1D()(cnn_repr_layer)
        residual_connection = Dense(cnn_dim, name='residual_seq_conn')(residual_connection)
        residual_connection = BatchNormalization()(residual_connection)
        residual_connection = Activation(activation='relu')(residual_connection)

        # Build Sequential layer

        seq_layer = self.sequential_layer(input=cnn_repr_layer, is_stateful=is_stateful)

        # Add residual connection to the sequential layer's output
        seq_residual_out = add([seq_layer, residual_connection])

        seq_residual_out = Dropout(0.3)(seq_residual_out)
        seq_residual_out = Dense(self.n_classes, activation='softmax', name='final_output')(seq_residual_out)

        seq_model = Model(inputs=input_signal, outputs=seq_residual_out)

        print seq_model.summary()

        if plot_arch:
            plot_model(seq_model, show_shapes=True, to_file="Seq_Arch.png")

        if finetune:
            # Load pretrained weights
            pretrained_model_path = os.path.join(self.MODEL_DIR, pretrain_model_name)
            pretrained_weights = os.path.join(pretrained_model_path, 'best_weights.hdf5')

            print "Loading weights from {}".format(pretrained_weights)
            self.logger.info("Loading weights from {}".format(pretrained_weights))
            seq_model.load_weights(filepath=pretrained_weights, by_name=True)

            optimizer = SGD(learning_rate)
            # gpu_seq_model = multi_gpu_model(model=seq_model, gpus=self.n_gpus, cpu_relocation=True)
            seq_model.compile(optimizer=optimizer, metrics=['accuracy'], loss=losses.categorical_crossentropy)

            return (None, seq_model)

        else:
            optimizer = Adam()
            # gpu_seq_model = multi_gpu_model(model=seq_model, gpus=self.n_gpus, cpu_relocation=True)
            seq_model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])

            return (None, seq_model)
