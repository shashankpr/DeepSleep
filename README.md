# DeepSleep

Master thesis on sleep pattern analysis and classification. Exploring the application of neural networks to model the time-dependent, sequential nature of sleep patterns from pressure-based signals.

## Dependencies Required

1. [TensorFlow][https://www.tensorflow.org/install/]

    Install using: `pip install tensorflow`

## Installation Instructions

This project is based on python 2.7 and uses [Keras][https://keras.io] with TensorFlow as the backend to deploy and train deep learning models.
Follow the steps to get the code running:
 1. Create virtual environment
    - If using Anaconda for python:

    `conda create -n yourenvname python=2.7 anaconda`

    - If using system Python :

    `virtualenv -p /usr/bin/python2.7 my_project`

 2. Activate the virtual environment:

     `source activate yourenvname`

     or

     `source my_project/bin/activate`

 3. Install the requirements:

    `pip install -r requirements.txt`

 4. Run code using:

    `python run.py`

## Tensorboard

To run tensorboard and visualize the network's change over the compilation:

`tensorboard --logdir=/full_path_to_your_logs`