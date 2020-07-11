#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add `Gaussian white noise`_ into the components of the ``Xtrain`` in the ``npz`` file with given SNR ratio.

We create Gaussian white noise signal into each component of ``Xtrain``, i.e., each column,
and save them in new folders in ``CASE_FOLDER`` naming ``CASE_NAME_noise_level_SNR_RATIO``.

Example:

    ::

        $ python make_recurrent_form_noisy_state.py

.. _Gaussian white noise:
    https://en.wikipedia.org/wiki/White_noise

.. _Signal-to-noise ratio:
    https://en.wikipedia.org/wiki/Signal-to-noise_ratio

Attributes:
    SNR_RATIO (:obj:`float`): Please refer to here for more details about `Signal-to-noise ratio`_.

    FIG_SIZE (:obj:`tuple` of :obj:`int`): Size of figure (width, height) saved in the new folder containing noisy data.

    PLT_CPNT_LIST (:obj:`list` of :obj:`int`): List of components to be plotted as illustration of the effect of noisy data.

    CASE_FOLDER (:obj:`str`): Path to the ``result`` folder of a ``case``.

    CASE_NAME (:obj:`str`): Case name. Moreover, ``CASE_NAME_noise_level_0`` is the folder containing the
        original data.

    DATA_WHOLE_STR (:obj:`str`): Filename of the whole trajectory of the raw data.

    DATA_TRAIN_STR (:obj:`str`): Filename of the training part of the raw data.

    DATA_TEST_STR (:obj:`str`): Filename of the testing part of the raw data.

"""

import numpy as np
from matplotlib import pyplot as plt
from PREP_DATA_SRC.source_code.lib.utilities import *

plt.style.use('siads')
sys.dont_write_bytecode = True
sys.path.insert(0, "../../../")

# We consider a given signal-to-noise ratio and choose which component to plot.

SNR_RATIO = 1
FIG_SIZE = (16, 4)
PLT_CPNT_LIST = [1, 14, 49]

# We set up the case folders, denoting where is the original training data
# and testing, whole data.

CASE_FOLDER = '../EXAMPLES/50d_cylinder_LRAN/result/'
CASE_NAME = '50d_cylinder_flow_pod_case'
DATA_WHOLE_STR = 'c6_re100_rank_50_POD_whole.npz'
DATA_TRAIN_STR = 'c6_re100_rank_50_POD_training.npz'
DATA_TEST_STR = 'c6_re100_rank_50_POD_testing.npz'

if __name__ == '__main__':

    # We read the original (noise-free) data from the above setup.

    FOLDER_NAME = CASE_FOLDER + CASE_NAME + '_noise_level_0'
    RAW_WHOLE_DATA = np.load(FOLDER_NAME + '/' + DATA_WHOLE_STR)
    RAW_TRAIN_DATA = np.load(FOLDER_NAME + '/' + DATA_TRAIN_STR)
    STATE = RAW_WHOLE_DATA['Xtrain']
    STATE_TRAIN = RAW_TRAIN_DATA['Xtrain']
    NUM_TOT_DATA = STATE.shape[0]
    NUM_TRAIN_DATA = STATE_TRAIN.shape[0]
    DIM_DATA = STATE.shape[1]

    # We create new folders for noisy data to be stored.

    DIR_NOISY = CASE_FOLDER + CASE_NAME + '_noise_level_' + str(SNR_RATIO)
    mkdir(DIR_NOISY)

    # We generate multi-variate normal (MVN) data followsing N(0,I) distribution,
    # using ``numpy.random.normal``.

    gaussian_random_0_1 = np.random.normal(size=STATE.shape)

    # Given ``SNR_RATIO``, we add MVN to original (noise-free) data to create noisy data.

    state_noise_array = []
    for index, s in enumerate(STATE):
        state_noise = s + SNR_RATIO / 100.0 * np.abs(s) * gaussian_random_0_1[index, :]
        state_noise_array.append(state_noise)

    # We transform the list containing noisy data into ``numpy.ndarray`` and
    # split the data into training and testing based on ``NUM_TRAIN_DATA``.

    state_noise_array = np.array(state_noise_array)
    state_noise_array_train = state_noise_array[:NUM_TRAIN_DATA, :]
    state_noise_array_test = state_noise_array[NUM_TRAIN_DATA:, :]

    # We plot trajectory for selected components given by ``PLT_CPNT_LIST``, with
    # figure size = ``FIG_SIZE`` and png is saved in ``DIR_NOISY``.

    for i_component in PLT_CPNT_LIST:
        plt.figure(figsize=FIG_SIZE)
        plt.plot(STATE[:, i_component], 'b.', alpha=0.5, label='true-1')
        plt.plot(state_noise_array[:, i_component], 'r.', alpha=0.5, label='noise-1')

        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig('./' + DIR_NOISY + '/component_plot_' + str(i_component) + ' .png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    # We save the whole data, training and testing data, into ``DIR_NOISY``.

    np.savez(DIR_NOISY + '/' + DATA_WHOLE_STR, Xtrain=state_noise_array)
    np.savez(DIR_NOISY + '/' + DATA_TRAIN_STR, Xtrain=state_noise_array_train)
    np.savez(DIR_NOISY + '/' + DATA_TEST_STR, Xtrain=state_noise_array_test)
