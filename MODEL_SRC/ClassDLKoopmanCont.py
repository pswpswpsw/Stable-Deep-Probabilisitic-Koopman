#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class of Continuous time Deep Learning Models for Koopman Operator"""

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "../")

# setup for Edward to be working
import os
import pprint
import L4
import cPickle as pickle

os.environ['TF_C_API_GRAPH_CONSTRUCTION'] = '0'

# GPU setup
import GPUtil

# import pprint
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.4, maxMemory=0.4)
    DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except:
    print('CPU mode')

import tensorflow as tf
import edward as ed
import shutil
import time
import numpy as np
import re

from lib.utilities import myswish_beta
from lib.utilities import penalized_tanh
from lib.utilities import matrix_exponential
from lib.utilities import mkdir
from lib.utilities import TF_FLOAT

from lib.conflux_version_sklearn import train_test_split
from lib.conflux_version_sklearn import StandardScaler
from numpy import linalg as LA
from edward.models import Normal
from edward.models import InverseGamma
from edward.models import PointMass
from edward.models import MixtureSameFamily
from edward.models import Categorical
from edward.models import Cauchy
from edward.models import NormalWithSoftplusScale
from edward.models import Gamma

from tensorflow.contrib.distributions import bijectors
from tensorflow.contrib.distributions import TransformedDistribution


class ClassDLKoopmanCont(object):
    """Class of a `deterministic` Deep learning model to find Koopman operators in a continuous time sense

    For more details, please refer to preprint_ .

    .. _preprint: https://arxiv.org/abs/1906.03663

    Args:
        configDict (:obj:`dict`): a dictionary contains model options of deterministic DL for continuous Koopman operator.

            Examples:

                ::

                    configDict = {
                        'caseName': '50d_cylinder_flow_pod_case_noise_level_'+str(noise_level),
                        "lr": 1e-4,
                        "structureEncoder": [50, 100, 50, 20, 20],
                        'numberEpoch': 30000,
                        'miniBatch': 128,
                        'decay': 1e-8,
                        'c_los_lin': 1.0,
                        'c_los_reg': 1e-10,
                        'typeRecon': 'nonlinear',
                        'phase_space_range': range_of_X,
                        'activation': 'swish',
                        'nsd': 1, # stable-K form.  #2: X-X^T -diag type  1:  efficient banded matrix - diag type  otherwise: not using anything
                        'normalization_type':'only_max',
                        'T': 100,
                        'dt': 0.1*2 # 2 is the coarse-graining factor
                    }

        edward_configDict (:obj:`dict`): a dictionary contains model options
            of bayesian DL.

            Examples:

                ::

                    edward_configDict = {
                        'ALPHA_HPP': 0,
                        'BETA_HPP': 1,
                        'sm2g': {'pi': 1.0 / 4.0,
                                 's1': 10 ** 0,
                                 's2': 10 ** -6},
                        'mode': 'ADVIARD',  # 'ADVIARD', 'MAPNoARD', 'MAPSimple', 'ADVInoARD',
                        'init': 'Standard',# 'nonBayes', # 'Standard'
                        'init_std_softplus': -8,
                        'MAP_dir': '../result/50d_cylinder_flow_pod_case_noise_level_0/dldmd_2019-04-02-17-07-31',
                        'verbose': False
                    }


        gpu_percentage (:obj:`float`): ratio for how much GPU memory to use.

    Attributes:
        model_params (:obj:`dict`): the dictionary that contains all deterministic DL setup.
            Besides, it adds two more as ``enable_edward``, and ``edward_cfg`` to indicate
            using edward or not.

        model_folder_name (:obj:`str`): name of the folder to put the saved model parameters. It is created
            by current machine time, starting with ``dldmd_``. If edward is enabled, it will add
            ``_Bayes`` as suffix.

        dir (:obj:`str`): the relative path to put the model into.

        initializer (:obj:`function`): an initializer (it is still a function).

        scale_initializer (:obj:`function`): an initializer for scale in variational posteriori.

        sess (:obj:`class`): A context manager using this session as the default session

        graph (:obj:`class`): A `Graph` instance supports an arbitrary number of "collections"
            that are identified by name.

        vp_dict (:obj:`dict`): a dict collection of variational posteriori distributions.

        hpp_dict (:obj:`dict`): a dict collection of hyperprior distribution.

        prior_dict (:obj:`dict`): a dict collection of priori distribution.

        K_X (:obj:`tf.Tensor`): Prob. distribution to construct stab. form.

        K_XX (:obj:`tf.Tensor`): anti-symmetric pro. distribution to construct stab. form.

        square_diagonal_line (:obj:`tf.Tensor`): the non-negative diagonal elements to construct stab. form.
            It needs to be act with   :obj:`tf.diag` to become really a matrix.

        koopmanOp_learn_intermediate (:obj:`tf.Tensor`): the construct tensor of Koopman operator K.

        koopmanOp_learn (:obj:`tf.Tensor`): the final tensor with name ``K`` of Koopman operator K.

        pi (:obj:`float`): Scale Mixture 2 Gaussian prior hyperparameter: :math:`\pi` in Google 2015 BNN paper: `at
            <http://arxiv.org/abs/1505.05424>`_.

        s1 (:obj:`float`): Scale Mixture 2 Gaussian prior hyperparameter: :math:`s_1` in Google 2015 BNN paper: `at
            <http://arxiv.org/abs/1505.05424>`_.

        s2 (:obj:`float`): Scale Mixture 2 Gaussian prior hyperparameter: :math:`s_2` in Google 2015 BNN paper: `at
            <http://arxiv.org/abs/1505.05424>`_.

        dimX (:obj:`int`): dimension of the input, i.e., system state :math:`x`.

        numKoopmanModes (:obj:`int`): dimension of the Koopman invariant observable space.

        activation (:obj:`function`): nonlinear activations in neural network.

        shape_list (:obj:`list`): [:attr:`numKoopmanModes`,:attr:`numKoopmanModes`]. This is the shape of Koopman
        operator K.

        encoderLayerWeights (:obj:`dict`): a dict collection of weights and biases for encoder.

        decoderLayerWegihts (:obj:`dict`): a dict collection of weights and biases for decoder.

        X (:obj:`tf.Tensor`): placeholder for the input states with shape as (None, :attr:`dimX`).

        eta (:obj:`tf.Tensor`): normalized input :math:`\eta = (X - \overline{X}) \Lambda^{-1}`,
            where :math:`\overline{X}` = :attr:`mu_X`, :math:`\Lambda^{-1}` = :attr:`Lambda_X_inv`.

        Xdot (:obj:`tf.Tensor`): placeholder for the input of time derivative as :math:`\dot{X}`.

        etaDot (:obj:`tf.Tensor`): normalized time derivative as :math:`\dot{\eta} = \dot{X} \Lambda^{-1}`.

        z_rec_loss (:obj:`tf.Tensor`): placeholder for reconstruction loss for Edward to build likelihood function.

        z_lin_loss (:obj:`tf.Tensor`): placeholder for linear dynamics loss for Edward to build likelihood function.

        koopmanPhi (:obj:`tf.Tensor`): Koopman observables.

        etaRec (:obj:`tf.Tensor`): reconstructed normalized states.

        Xrec (:obj:`tf.Tensor`): reconstructed states as :math:`\overline{X} + \eta \Lambda`.

        rec_loss (:obj:`tf.Tensor`): mean square error between :attr:`X` and :attr:`Xrec`.

        norm_rec_loss (:obj:`tf.Tensor`): MSE between :attr:`eta` and :attr:`etaRec`.

        fdphidx (:obj:`tf.Tensor`): :math:`\dot{\eta} \cdot \partial \phi /\partial X`.

        kphi (:obj:`tf.Tensor`): :math:`\phi K`.

        lin_loss (:obj:`tf.Tensor`): MSE between :attr:`kphi` and :attr:`fdphidx`.

        reg_parameter_loss (:obj:`tf.Tensor`): regularization parameter loss with biases and weights.

        loss_op (:obj:`tf.Tensor`): total loss objective function.

        optimizer (:obj:`class`): optimizer class instance.

        train_op (:obj:`function`) a function that updates variables in ``var_list``,. i.e., the backpropagation
            algorithm.

        Xtrain (:obj:`numpy.ndarray`): training states :math:`X`.

        XdotTrain (:obj:`numpy.ndarray`): training time derivative :math:`\dot{X}`.

        Xvalid (:obj:`numpy.ndarray`): valid states :math:`X`.

        XdotValid (:obj:`numpy.ndarray`): valid time derivative :math:`\dot{X}`.

        mu_X (:obj:`tf.Tensor`): sampled mean of training states.

        Lambda_X (:obj:`tf.Tensor`): sampled mean of training time derivative.

        loss_dict (:obj:`dict`): a collection of training loss to be saved.

        K_sample (:obj:`numpy.ndarray`): array of Monte Carlof sampling of ``K`` matrix.

    """

    def __init__(self, configDict, edward_configDict=None, gpu_percentage=0.05):

        # Define deterministic DL model parameters

        self.model_params = {'caseName':           configDict['caseName'],
                             "learningRate":       configDict['lr'],
                             "structureEncoder":   configDict['structureEncoder'],
                             'numberEpoch':        configDict['numberEpoch'],
                             'miniBatch':          configDict['miniBatch'],
                             'decay':              configDict['decay'],
                             'c_los_lin':          configDict['c_los_lin'],
                             'c_los_reg':          configDict['c_los_reg'],
                             'typeRecon':          configDict['typeRecon'],
                             'phase_space_range':  configDict['phase_space_range'],
                             'activation':         configDict['activation'],
                             'nsd':                configDict['nsd'],
                             'normalization_type': configDict['normalization_type'],
                             'SVD_DMD':            configDict['SVD_DMD'] # if not defined, we put true in it
                             }

        # We pass edward configurations into :attr:`model_params`.

        if edward_configDict != None:

            self.model_params['enable_edward'] = True
            self.model_params['edward_cfg'] = edward_configDict

        else:

            self.model_params['enable_edward'] = False

        # We build directory for result with time stamp.

        ts = time.gmtime()
        self.model_folder_name = 'dldmd_' + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
        if edward_configDict != None:
            self.model_folder_name = self.model_folder_name + '_Bayes'
        self.dir = '../result/' + self.model_params['caseName'] + '/' + self.model_folder_name


        # We make directory for result and the specific case.

        mkdir(directory='../result')
        mkdir(directory=self.dir)


        # write configuration into dir
        with open(self.dir + '/cfg_koopman_cont.txt', 'w') as file:
            file.write('summary of configuration of Koopman learning: \n')
            for key in self.model_params:
                file.write('--\n')
                file.write('key = ' + str(key) + '\n')
                file.write('value = ' + str(self.model_params[key]) + '\n\n')

        # We clean directory of previous tensorboard if there is

        directory = self.dir + '/tbTrain'
        if os.path.exists(directory):
            shutil.rmtree(directory)
        directory = self.dir + '/tbValid'
        if os.path.exists(directory):
            shutil.rmtree(directory)

        # We choose one init for weights and biases

        # - use variance scaling initilizer.
        # self.initializer = tf.contrib.layers.variance_scaling_initializer(dtype=TF_FLOAT)

        # - use Xaiver initializer or He initializer.
        self.initializer = tf.contrib.layers.xavier_initializer(dtype=TF_FLOAT)

        # We setup the init way for the scale as constant initializer with given ``init_std_softplus``.
        if edward_configDict == None:
            pass
        else:
            self.scale_initializer = tf.constant_initializer(edward_configDict['init_std_softplus'], dtype=TF_FLOAT)

        # We initialize a session throughout the class with given GPU memory resources

        try:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_percentage)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        except:
            self.sess = tf.Session()

        # We asign the default graph of that session to the graph of the class

        self.graph = self.sess.graph

        # Number of samples for sampling K
        self.N_SAMPLES = 100


    def _create_half_cauchy_prior(self, loc_tensor, scale_tensor):
        """Create half cauchy distribution with ``edward.models.TransformedDistribution``
        together with ``edward.models.Cauchy``.

        This is recommended by Andrew Gelman at `here`_.
        More information about `half cauchy`_.

        .. _here: http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf
        .. _half cauchy: https://en.wikipedia.org/wiki/Cauchy_distribution

        Args:
            loc_tensor (:obj:`tf.Tensor`): location of the cauchy distribution.
            scale_tensor (:obj:`tf.Tensor`): scale of the cauchy distribution.

        Returns:
            :obj:`tf.Tensor` : Half-Cauchy distribution.

        """

        dist = Cauchy(loc=loc_tensor, scale=scale_tensor)
        return ed.models.TransformedDistribution(distribution=dist, bijector=bijectors.AbsoluteValue())

    def _create_scale_log_normal_0_1_prior(self, shape_list):
        """Create LogNormal(0,1) as prior given ``shape_list``.

        Args:
            shape_list (:obj:`list`): the shape of the MV distributions.

        Returns:
            :obj:`tf.Tensor` : `shape_list` * LogNormal(0,1) distribution

        """
        return ed.models.TransformedDistribution(
            distribution=Normal(loc=tf.zeros(shape_list, dtype=TF_FLOAT), scale=tf.ones(shape_list, dtype=TF_FLOAT)),
            bijector=bijectors.Exp())

    def _create_scale_log_normal_0_given_scale(self, shape_list, scale_tensor, object_name):
        """Create LogNormal(0, ``scale_tensor``) distribution.

        Args:
            shape_list (:obj:`list`): the shape of MV distributions.

            scale_tensor (:obj:`tf.Tensor`): this is the scale for the LogNormal distributions

            object_name (:obj:`str`): the name for the transformed distribution.

        Returns:
            :obj:`tf.Tensor` : `shape_list` * LogNormal(0, `scale_tensor` ) distribution.

        """
        return ed.models.TransformedDistribution(
            distribution=Normal(loc=tf.zeros(shape_list, dtype=TF_FLOAT), scale=scale_tensor), bijector=bijectors.Exp(),
            name=object_name)

    def _create_gamma_chi2_given_alpha(self, shape_list, scale_tensor, object_name):
        """Create ChiSquare(0, ``scale_tensor``) distribution

        We create ChiSquare distribution from Gamma distribution using certain
        alpha, which is just the scale.

        Args:
            shape_list (:obj:`list`): the shape of MV distributions.

            scale_tensor (:obj:`tf.Tensor`): this is the scale for the ChiSquare distributions

            object_name (:obj:`str`): the name for the transformed distribution.

        Returns:
            :obj:`tf.Tensor` : `shape_list` * ChiSquare(0, `scale_tensor` ) distribution.

        """

        rate = tf.constant([0.5], dtype=TF_FLOAT) * tf.ones(shape_list, dtype=TF_FLOAT)
        rate_cast = tf.cast(rate, TF_FLOAT)
        return ed.models.Gamma(concentration=scale_tensor, rate=rate_cast, name=object_name)

    def _create_scale_log_normal(self, shape_list, loc_name, scale_name, object_name):
        """Create LogNormal(``variable_loc``, ``variable_scale``) distribution for variational posterior.

        Since it is for VP, so the location and scale are all created using :obj:`tf.get_variable`.

        Args:
            shape_list (:obj:`list`): the shape of MV distributions.

            loc_name (:obj:`str`): variable name for the locations

            scale_name (:obj:`str`): variable name for the scale

            object_name (:obj:`str`): the name for the transformed distribution. Note that
                the true name contains an affix `q`.

        Returns:
            :obj:`tf.Tensor` : `shape_list` * LogNormal(0, `scale_tensor` ) distribution.

        """

        return ed.models.TransformedDistribution(distribution=ed.models.NormalWithSoftplusScale(
            loc=tf.get_variable(loc_name, shape_list, initializer=tf.constant_initializer(-9.)),
            scale=tf.get_variable(scale_name, shape_list, initializer=tf.constant_initializer(-9.))),
            bijector=bijectors.Exp(), name='q' + object_name)

    def _create_standard_normal(self, shape_list, loc_name, scale_name, object_name):
        """Create standard normal(``variable_loc``, ``variable_scale``) distribution for variational posterior.

        Since it is for VP, so the location and scale are all created using :obj:`tf.get_variable`.

        Args:
            shape_list (:obj:`list`): the shape of MV distributions.

            loc_name (:obj:`str`): variable name for the locations

            scale_name (:obj:`str`): variable name for the scale

            object_name (:obj:`str`): the name for the transformed distribution. Note that
                the true name contains an affix `q`.

        Returns:
            :obj:`tf.Tensor` : `shape_list` * Normal(0, `scale_tensor` ) distribution.

        """

        return ed.models.NormalWithSoftplusScale(
            loc=tf.get_variable(loc_name, shape_list, initializer=self.initializer),
            scale=tf.get_variable(scale_name, shape_list, initializer=self.scale_initializer), name='q' + object_name)

    def _create_sm2g_prior(self, shape_list, name):
        """Create scale mixture prior of two Gaussian distributions.

        For more information, refer to sec 3.3 in Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
        Weight Uncertainty in Neural Networks. 37. Retrieved `at <http://arxiv.org/abs/1505.05424>`_.

        Note:
            :attr:`model_params` contains `edward_cfg.sm2g` of `pi`, `s1`, `s2` which determine the prior distribution.

        Args:
            shape_list (:obj:`list`): the shape of MV distributions.

            name (:obj:`str`): the name for the transformed distribution.

        Returns:
            :obj:`tf.Tensor` : scaled mixture prior distribution.

        """
        shape_list_local = shape_list[:]  # ! copy without reference
        shape_list_local.append(1)
        shape = tuple(shape_list_local)
        gm = MixtureSameFamily(mixture_distribution=Categorical(probs=tf.ones(shape) * [self.model_params['edward_cfg']['sm2g']['pi'],
                                    1.0 - self.model_params['edward_cfg']['sm2g']['pi']]), components_distribution=Normal(loc=tf.ones(shape) * [0., 0.], # component 2
                                           scale=tf.ones(shape) * [self.model_params['edward_cfg']['sm2g']['s1'],
                                               self.model_params['edward_cfg']['sm2g']['s2']]), name=name)
        return gm

    def setup_K(self):
        """Setup for the Koopman operator K matrix.

        We will first check if the case is Bayesian or not. Then we will use the stabilization form, almost as always.
        The stabilization form can be found in our `preprint`_.

        .. _preprint: https://arxiv.org/abs/1906.03663

        Note:

            It contains three aspects:

            1. variational posteriori

            2. hyperprior

            3. prior

        """

        self.vp_dict = {}
        self.hpp_dict = {}
        self.prior_dict = {}

        if self.model_params['enable_edward']:

            # We setup the K for: Bayes DL case.

            if self.model_params['edward_cfg']['mode'] == 'MAPSimple':

                # Maximum aposteriori with simple variance setup.

                if self.model_params['nsd'] == 1:

                    # Stabilization: K = banded anti-symmetri matrix - diag(a_1^2,..,a_n^2),
                    # Note that prior doesn't have square at all. So just a positive range prior
                    #  is enough.

                    print 'nsd = anti-symmetric-banded - diag enabled'

                    # K_X is just random square matrix, but will be truncated only for first offset
                    # diagonal prior for K_X = Normal(0,1)

                    self.K_X = Normal(loc=tf.zeros(self.shape_list, dtype=TF_FLOAT),
                                      scale=tf.ones(self.shape_list, dtype=TF_FLOAT))
                    X_upper = tf.matrix_band_part(self.K_X, 0, 1)
                    self.K_XX = X_upper - tf.transpose(X_upper)

                    # We create half cauchy(0,1) for the priori, so it is positive

                    self.square_diagonal_line = self._create_half_cauchy_prior(tf.zeros(self.shape_list[0], dtype=TF_FLOAT),
                                                                               tf.ones(self.shape_list[0], dtype=TF_FLOAT))

                    diagonal_square_matrix = tf.diag(self.square_diagonal_line)

                    # Stabilization: K = banded anti-symmetri matrix - diag(a_1^2,..,a_n^2)

                    self.koopmanOp_learn_intermediate = self.K_XX - diagonal_square_matrix
                    self.koopmanOp_learn = tf.identity(self.koopmanOp_learn_intermediate, name='K')  # name the Koopman tensor with name 'K'

                    # We save priori for NSD components of Koopman K in the prior_dict

                    self.prior_dict['K_K_X'] = self.K_X
                    self.prior_dict['K_SD'] = self.square_diagonal_line

                    # We setup posteriori for NSD components of Koopman K using `PointMass` so as to match

                    self.vp_dict['K_K_X'] = PointMass(
                        params=tf.get_variable("qK_K_X/loc", self.shape_list, 
                                               initializer=self.initializer,
                                               dtype=TF_FLOAT))
                    self.vp_dict['K_SD'] = PointMass(
                        params=tf.get_variable("qK_SD/loc", self.shape_list, 
                                               initializer=self.initializer,
                                               dtype=TF_FLOAT))

                else:

                    # No stabilization, we just use point mass and K is just a simple pointmass

                    # Prior just a Normal

                    self.koopmanOp_learn = Normal(loc=tf.zeros(self.shape_list, dtype=TF_FLOAT),
                                                  scale=tf.ones(self.shape_list, dtype=TF_FLOAT), name='K')
                    self.prior_dict['K'] = self.koopmanOp_learn

                    # Posteriori for Koopman K
                    self.vp_dict['K'] = PointMass(params=tf.get_variable("qK/loc", self.shape_list, 
                                                                         initializer=self.initializer, 
                                                                         dtype=TF_FLOAT))

            elif self.model_params['edward_cfg']['mode'] == 'ADVInoARD':

                # With standard ADVI but prior is fixed.

                if self.model_params['nsd'] == 1:

                    # Stabilization: K = banded anti-symmetri matrix - diag(a_1^2,..,a_n^2) """

                    print 'nsd = anti-symmetric-banded - diag is enabled'

                    # :attr:`K_X` : prior for K_X with sm2g prior

                    self.K_X = self._create_sm2g_prior(self.shape_list,name='K')  # Normal(loc=tf.zeros(self.shape_list), scale=tf.ones(self.shape_list))

                    # We take the off-diagonal part
                    X_upper = tf.matrix_band_part(self.K_X, 0, 1)
                    self.K_XX = X_upper - tf.transpose(X_upper)

                    # We take prior for square of diagonal line is HalfCauchy(0,1)
                    self.square_diagonal_line = self._create_half_cauchy_prior(tf.zeros(self.shape_list[0], dtype=TF_FLOAT),
                                                                               tf.ones(self.shape_list[0], dtype=TF_FLOAT))

                    # Alternative is to take logNormal
                    # self.square_diagonal_line = self._create_scale_log_normal_0_1_prior([self.shape_list[0], ])

                    diagonal_square_matrix = tf.diag(self.square_diagonal_line)

                    self.koopmanOp_learn_intermediate = self.K_XX - diagonal_square_matrix
                    self.koopmanOp_learn = tf.identity(self.koopmanOp_learn_intermediate, name='K')

                    # We assign priori for the NSD components of K

                    self.prior_dict['K_K_X'] = self.K_X
                    self.prior_dict['K_SD'] = self.square_diagonal_line

                    # We define posteriori for NSD components of Koopman K

                    self.vp_dict['K_K_X'] = self._create_standard_normal(self.shape_list, 'qK_K_X/loc', 'qK_K_X/scale', 'K_K_X')
                    self.vp_dict['K_SD'] = self._create_scale_log_normal([self.shape_list[0], ], 'qK_SD/loc', 'qK_SD/scale', 'K_SD')

                else:

                    # We simply use Normal for K if there is no stabilization.

                    # self.pi = self.model_params['edward_cfg']['sm2g']['pi']
                    # self.s1 = self.model_params['edward_cfg']['sm2g']['s1']
                    # self.s2 = self.model_params['edward_cfg']['sm2g']['s2']

                    # We take prior as scale mixture two Gaussian

                    self.koopmanOp_learn = self._create_sm2g_prior(self.shape_list, name='K')
                    self.prior_dict['K'] = self.koopmanOp_learn

                    # We assign posteriori as standard normal for K.

                    self.vp_dict['K'] = self._create_standard_normal(self.shape_list, "qK/loc", "qK/scale", "K")


            elif self.model_params['edward_cfg']['mode'] == 'ADVIARD':

                # With scale also as optimized parameters, we enable hyperpriors for model parameters

                self.ALPHA_HPP = self.model_params['edward_cfg']['ALPHA_HPP']
                self.BETA_HPP = self.model_params['edward_cfg']['BETA_HPP']

                if self.model_params['nsd'] == 1:

                    # stabilization K = banded anti-symmetric matrix - diag(a_1^2,..,a_n^2)

                    print 'nsd = anti-symmetric-banded - diag enabled'

                    # We set up the HyperPrior for the square of scale for the prior K_X and SD
                    # HalfCauchy(alpha_hpp, beta_hpp)

                    self.hpp_dict['K_K_X'] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(self.shape_list, dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(self.shape_list, dtype=TF_FLOAT))
                    self.hpp_dict['K_SD'] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(self.shape_list[0], dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(self.shape_list[0], dtype=TF_FLOAT))

                    # Alternative is to use InverseGamma
                    # self.hpp_dict['K_K_X'] = InverseGamma([self.ALPHA_HPP] * tf.ones(self.shape_list, dtype=TF_FLOAT),
                    #                                       [self.BETA_HPP] * tf.ones(self.shape_list,dtype=TF_FLOAT))
                    # self.hpp_dict['K_SD'] = InverseGamma([self.ALPHA_HPP] * tf.ones(self.shape_list[0], dtype=TF_FLOAT),
                    #                                      [self.BETA_HPP] * tf.ones(self.shape_list[0],dtype=TF_FLOAT))

                    # prior for K_X is N(0, sigma), so sigma is determined from a Hierarchical Bayesian..self.hpp_dict['K_K_X']

                    self.K_X = Normal(loc=tf.zeros(self.shape_list, dtype=TF_FLOAT),
                                      scale=tf.sqrt(self.hpp_dict['K_K_X']), name='K_X')
                    X_upper = tf.matrix_band_part(self.K_X, 0, 1)
                    self.K_XX = X_upper - tf.transpose(X_upper)

                    # prior for K_SD as log normal distribution
                    # self.square_diagonal_line = self._create_scale_log_normal_0_given_scale([self.shape_list[0], ], tf.sqrt(self.hpp_dict['K_SD']), 'K_SD')

                    # prior for K_SD is a Gamma(concentration, 0.5), concentration is determined by a Hierarchical Bayesian...self.hpp_dict['K_SD']
                    self.square_diagonal_line = self._create_gamma_chi2_given_alpha([self.shape_list[0], ],
                        self.hpp_dict['K_SD'], 'K_SD')

                    # self.square_diagonal_line = self._create_scale_log_normal_0_given_scale([self.shape_list[0], ], tf.sqrt(self.hpp_dict['K_SD']), 'K_SD')
                    diagonal_square_matrix = tf.diag(self.square_diagonal_line, name='SD')

                    self.koopmanOp_learn_intermediate = self.K_XX - diagonal_square_matrix
                    self.koopmanOp_learn = tf.identity(self.koopmanOp_learn_intermediate, name='K')

                    # We store priori for NSD components of Koopman K

                    self.prior_dict['K_K_X'] = self.K_X
                    self.prior_dict['K_SD'] = self.square_diagonal_line

                    # We define posteriori for NSD components of Koopman K

                    self.vp_dict['K_K_X'] = self._create_standard_normal(self.shape_list, 'qK_K_X/loc', 'qK_K_X/scale',
                                                                         'K_K_X')
                    self.vp_dict['K_SD'] = self._create_scale_log_normal([self.shape_list[0], ], 'qK_SD/loc',
                                                                         'qK_SD/scale', 'K_SD')

                    # We define the variational posteriori for the prior variance of NSD Components of Koopman K

                    self.vp_dict['scale_K_K_X'] = self._create_scale_log_normal(self.shape_list, "qscaleK_K_X/loc",
                                                                                "qscaleK_K_X/scale", "qscaleK_K_X")
                    self.vp_dict['scale_K_SD'] = self._create_scale_log_normal([self.shape_list[0], ], "qscaleK_SD/loc",
                        "qscaleK_SD/scale", "qscaleK_SD")


                else:

                    ### no stabilization

                    # self.hpp_dict['K'] = InverseGamma([self.ALPHA_HPP] * tf.ones(self.shape_list),
                    #                                   [self.BETA_HPP] * tf.ones(self.shape_list), name='scale_K')
                    self.hpp_dict['K'] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(self.shape_list, dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(self.shape_list, dtype=TF_FLOAT))

                    #### define the priors for Koopman K
                    self.koopmanOp_learn = Normal(loc=tf.zeros(self.shape_list, dtype=TF_FLOAT),
                                                  scale=tf.sqrt(self.hpp_dict['K']), name='K')
                    self.prior_dict['K'] = self.koopmanOp_learn

                    ### define the variational posteriori for Koopman K
                    self.vp_dict['K'] = self._create_standard_normal(self.shape_list, "qK/loc", "qK/scale", "K")

                    ### define the variational posteriori for the prior variance of Koopman K

                    self.vp_dict['scale_K'] = self._create_scale_log_normal(self.shape_list, "qscaleK/loc",
                                                                            "qscaleK/scale", "qscaleK")

            else:

                # shouldn't come to this line! error!

                raise NotImplementedError

        else:

            # If deterministic DL Koopman mode, then we don't use Bayesian, just tf.Variable.

            if self.model_params['nsd'] == 1:

                # K = banded anti-symmetri matrix - diag(a_1^2,..,a_n^2)

                print 'stabilization used: anti-symmetric-banded - diag enabled'

                self.K_X = tf.Variable(self.initializer(self.shape_list), name='K_K_X', dtype=TF_FLOAT)
                X_upper = tf.matrix_band_part(self.K_X, 0, 1)
                self.K_XX = X_upper - tf.transpose(X_upper)

                diagonal_line = tf.Variable(self.initializer([self.shape_list[0], ]), dtype=TF_FLOAT)
                square_diagonal_line = tf.square(diagonal_line,
                                                 name='K_SD')  # so K_SD only corresponds to the diagonal part, positive
                diagonal_square_matrix = tf.diag(square_diagonal_line)

                self.koopmanOp_learn_intermediate = self.K_XX - diagonal_square_matrix
                self.koopmanOp_learn = tf.identity(self.koopmanOp_learn_intermediate, name='K')

            elif self.model_params['nsd'] == 2:

                # K = banded anti-symmetri matrix - diag( abs a_1,.. ,abs a_n)

                print 'stabilization used: anti-symmetric-banded - diag enabled: but with abs'

                self.K_X = tf.Variable(self.initializer(self.shape_list), name='K_K_X', dtype=TF_FLOAT)
                X_upper = tf.matrix_band_part(self.K_X, 0, 1)
                self.K_XX = X_upper - tf.transpose(X_upper)

                diagonal_line = tf.Variable(self.initializer([self.shape_list[0], ]), dtype=TF_FLOAT)
                square_diagonal_line = tf.abs(diagonal_line, name='K_SD')
                diagonal_square_matrix = tf.diag(square_diagonal_line)

                self.koopmanOp_learn_intermediate = self.K_XX - diagonal_square_matrix
                self.koopmanOp_learn = tf.identity(self.koopmanOp_learn_intermediate, name='K')

            else:

                print 'no stabilization is used!'
                self.koopmanOp_learn = tf.Variable(self.initializer(self.shape_list), name='K', dtype=TF_FLOAT)

    def _set_dimension_and_activations(self):
        """Set the dimension for the Koopman observables and activations"""

        # We define dimension of input and number of koopman modes

        self.dimX = self.model_params['structureEncoder'][0]
        self.numKoopmanModes = self.model_params['structureEncoder'][-1]

        # We define activation function

        if self.model_params['activation'] == 'tanh':

            self.activation = tf.nn.tanh

        elif self.model_params['activation'] == 'ptanh':

            self.activation = penalized_tanh

        elif self.model_params['activation'] == 'elu':

            self.activation = tf.nn.elu

        elif self.model_params['activation'] == 'relu':

            self.activation = tf.nn.relu

        elif self.model_params['activation'] == 'selu':

            self.activation = tf.nn.selu

        elif self.model_params['activation'] == 'swish':

            self.activation = myswish_beta

        elif self.model_params['activation'] == 'sin':

            self.activation = tf.sin

        else:

            raise NotImplementedError('activation not implemented!')

        # We define the shape list for Koopman operator

        self.shape_list = [self.numKoopmanModes, self.numKoopmanModes]

    def _create_encoder_decoder_and_pod_weights(self):
        """Create the weights and biases for encoder and  decoders"""

        # We define layers' weights & biases of encoder

        # self.encoderLayerWeights, self.encoderLayerBias = self.buildMLPWeightsBias(
        #     structure_list=self.model_params['structureEncoder'], prefix='enc')

        self.encoderLayerWeights, self.encoderLayerBias = self.buildSpecialMLPWeightBias(
            structure_list=self.model_params['structureEncoder'], prefix='enc')

        # We define layers' weights & biases of decoder

        # self.decoderLayerWeights, self.decoderLayerBias = self.buildMLPWeightsBias(
        #     structure_list=self.model_params['structureEncoder'][::-1], prefix='dec')

        if self.model_params['typeRecon'] == 'nonlinear':
            self.decoderLayerWeights, self.decoderLayerBias = self.buildSpecialMLPWeightBias(
                structure_list=self.model_params['structureEncoder'][::-1], prefix='dec')
        else:
            if self.model_params['SVD_DMD']:
                print('build linear weights/biases for decoder')
                structureDecoder = [self.model_params['structureEncoder'][-1],
                                    self.model_params['structureEncoder'][-2],
                                    self.model_params['structureEncoder'][0]]
            else:
                print('build linear weights/biases for decoder')
                structureDecoder = [self.model_params['structureEncoder'][-1],
                                    self.model_params['structureEncoder'][0]]

            self.decoderLayerWeights, self.decoderLayerBias = self.buildSpecialMLPWeightBias(
                structure_list=structureDecoder, prefix='dec')

        # Define encoder POD weights
        self.encoderPODWeights = self.buildPODWeights(
            structure_list=self.model_params['structureEncoder'],
            prefix='enc')

        # Define decoder POD weights
        self.decoderPODWeights = self.buildPODWeights(
            structure_list=self.model_params['structureEncoder'][::-1],
            prefix='dec')

    def encoding_neural_net_with_pod(self, input, SVD_DMD_FLAG):
        """Create Koopman observables with POD short cut or not.

        Depending on :attr:`model_params`, we will encoder that with POD or not.

        Args:
            input (:obj:`tf.Tensor`): input of the neural network, i.e., system state :math:`x`.

        Returns:
            :obj:`tf.Tensor` : Koopman observables.

        """

        # feedforward NN encoder without the last weight (note last layer is also without activation)
        numWeightedLayers = len(self.encoderLayerWeights)

        if SVD_DMD_FLAG:

            output_NN = self.createMLPOutLinear(start_layer=input,
                                                weights=self.encoderLayerWeights,
                                                bias=self.encoderLayerBias,
                                                prefix='enc')

            output_svd = tf.matmul(input, self.encoderPODWeights['enc_POD'])
            phi_before_wm = output_NN + output_svd

            # get last output with weight muplication only, no activation
            phi = tf.matmul(phi_before_wm, self.encoderLayerWeights['enc_w_' + str(numWeightedLayers)])

        else:

            output_NN = self.createMLPOutLinearOLD(start_layer=input,
                                                weights=self.encoderLayerWeights,
                                                bias=self.encoderLayerBias,
                                                prefix='enc')

            phi_before_wm = output_NN
            phi = phi_before_wm


        return phi

    def decoding_neural_net_with_pod(self, phi, SVD_DMD_FLAG):
        """Output the state given Kopman observables

        Args:
            phi (:obj:`tf.Tensor`): Koopman observables.

        Returns:
            :obj:`tf.Tensor` : reconstructed system state.

        """


        # third, depending on if SVD shortcut is enabled or not
        if SVD_DMD_FLAG:
            # first, take phi with a pure weight multiplication
            phi_after_wm = tf.matmul(phi, self.decoderLayerWeights['dec_w_1'])

            # second, take that into a standard Feedforward NN
            psi_phi_NN = self.createMLPOutLinear(start_layer=phi_after_wm,
                                                 weights=self.decoderLayerWeights,
                                                 bias=self.decoderLayerBias,
                                                 prefix='dec')

            psi_svd = tf.matmul(phi_after_wm, self.decoderPODWeights['dec_POD'])
            eta_rec = psi_phi_NN + psi_svd
        else:
            phi_after_wm = phi
            psi_phi_NN = self.createMLPOutLinearOLD(start_layer=phi_after_wm,
                                                 weights=self.decoderLayerWeights,
                                                 bias=self.decoderLayerBias,
                                                 prefix='dec')

            eta_rec = psi_phi_NN

        return eta_rec

    def _compute_ddt_scalar(self, scalar_function, f):
        """Compute the dS/dt.

        Following the equation: :math:`\dot{\eta} \cdot \partial S/ \partial \eta`

        Args:
            scalar_function (:obj:`tf.Tensor`): the function to be taken ddt.

            f (:obj:`tf.Tensor`): it is f = :attr:`etaDot`.

        Returns:

            :obj:`tf.Tensor` : dS/dt.

        """
        dscalardx = tf.gradients(scalar_function, self.eta)[0]  # note: dphidx_i[0] is because the return is a list of one element
        tmp_dotproduct = tf.multiply(dscalardx, f)
        return tf.reduce_sum(tmp_dotproduct, axis=1)

    def _compute_f_nabla_koopman_phi(self, f):
        """Compute :math:`f \cdot \partial \phi / \partial x`

        Args:
            f (:obj:`tf.Tensor`): it is f = :attr:`etaDot`.

        Returns:
            :obj:`tf.Tensor` : :math:`f \cdot \partial \phi / \partial x`

        """
        fdphidx_list = []

        # We compute each d component in phi / dt

        for index in xrange(self.koopmanPhi.shape[1]):
            fdphidx_list.append(self._compute_ddt_scalar(scalar_function=self.koopmanPhi[:, index], f=f))

        # Stack dphi/dx for all phi components

        return tf.stack(fdphidx_list, axis=1)

    def construct_model(self):
        """Construct the main model graph for DL Koopman

        Returns:
            :obj:`tuple` : (residual_vector_rec_loss, residual_vector_lin_loss)

        """

        # Setup dimension and activations

        self._set_dimension_and_activations()

        # Data input placeholder
        # input of the neural network as a SINGLE state X
        self.X = tf.placeholder(TF_FLOAT, [None, self.dimX], name='X')

        # Transform into eta: normalized space.
        self.eta = tf.matmul(self.X - self.mu_X, self.Lambda_X_inv)

        # input of the neural network with state derivative Xdot
        self.Xdot = tf.placeholder(TF_FLOAT, [None, self.dimX], name='Xdot')

        # Transform into deta/dt
        self.etaDot = tf.matmul(self.Xdot, self.Lambda_X_inv)

        # Placeholder for Bayesian likelihood functions
        self.z_rec_loss = tf.placeholder(TF_FLOAT, [None, self.dimX], name='Z_rec_loss')
        self.z_lin_loss = tf.placeholder(TF_FLOAT, [None, self.numKoopmanModes], name='Z_lin_loss')

        # Setup K
        self.setup_K()

        # Setup encoder & decoder weights
        # note that zero-padding in POD_V is also performed here
        self._create_encoder_decoder_and_pod_weights()

        # Define the network topology
        # Get phi
        self.koopmanPhi = self.encoding_neural_net_with_pod(input=self.eta,
                                                            SVD_DMD_FLAG=self.model_params['SVD_DMD'])
        self.koopmanPhi = tf.identity(self.koopmanPhi, name='phi')

        # decoding with phi_nn and phi_dmd
        self.etaRec = self.decoding_neural_net_with_pod(phi=self.koopmanPhi,
                                                        SVD_DMD_FLAG=self.model_params['SVD_DMD'])

        # scaling back to X
        self.Xrec = tf.add(self.mu_X, tf.matmul(self.etaRec, self.Lambda_X), name='Xrec')

        # build loss

        normConst = tf.constant(1.0, dtype=TF_FLOAT) / (
                tf.constant(1.0, dtype=TF_FLOAT) + self.model_params['c_los_lin'] + self.model_params['c_los_reg'])

        # setup Reconstruction loss
        # unnormalized loss: have preference over mag

        self.rec_loss = tf.reduce_mean(tf.squared_difference(self.X, self.Xrec))

        # get normalized loss

        self.norm_rec_loss = tf.reduce_mean(tf.squared_difference(self.etaRec, self.eta))

        # compute linear dynamics loss

        self.fdphidx = self._compute_f_nabla_koopman_phi(f=self.etaDot)

        # ## compute dot F dot product phi, by summing things up
        # print 'koopmanOp_learn shape:',self.koopmanOp_learn.get_shape()
        # print 'koopmanPhi shape:',self.koopmanPhi.get_shape()
        # print 'fdphidx shape:',self.fdphidx.get_shape()

        # computing dphi/dot after koopman K on phi

        self.kphi = tf.matmul(self.koopmanPhi, self.koopmanOp_learn)

        ## finally add linear dynamics loss

        # no normalization
        self.lin_loss = tf.reduce_mean(tf.squared_difference(self.kphi, self.fdphidx))
        # some normalization
        # self.lin_loss = tf.reduce_mean(tf.squared_difference(self.kphi, self.fdphidx)/(tf.linalg.norm(self.kphi)+1e-6))

        # weight reg. loss
        # Weight-decay on encoder & encoder weights

        encoderW_l2_sum_loss = sum([tf.nn.l2_loss(weight) for weight in self.encoderLayerWeights.values()])
        decoderW_l2_sum_loss = sum([tf.nn.l2_loss(weight) for weight in self.decoderLayerWeights.values()])

        self.reg_parameter_loss = encoderW_l2_sum_loss + decoderW_l2_sum_loss

        ## add weight decay also to the biases

        encoderB_l2_sum_loss = sum([tf.nn.l2_loss(bias) for bias in self.encoderLayerBias.values()])
        decoderB_l2_sum_loss = sum([tf.nn.l2_loss(bias) for bias in self.decoderLayerBias.values()])

        ## regularization loss is a sum of weights and biases contributions

        self.reg_parameter_loss = self.reg_parameter_loss + encoderB_l2_sum_loss + decoderB_l2_sum_loss

        ## also regularize K
        self.reg_parameter_loss = self.reg_parameter_loss + tf.nn.l2_loss(self.koopmanOp_learn)

        # 4. Summing up losses
        self.loss_op = self.norm_rec_loss + self.model_params['c_los_lin'] * self.lin_loss \
                       + self.model_params['c_los_reg'] * self.reg_parameter_loss

        # normalized cost function...
        self.loss_op = normConst * self.loss_op

        # return the mean-eta-rec loss and mean-linear-eta loss
        # return prior for the hpp of W,b,K
        # return the prior for W,b,K

        # obtain the residual vector of reconstruction

        residual_vector_rec_loss = self.etaRec - self.eta
        residual_vector_rec_loss = tf.identity(residual_vector_rec_loss, name='residual_rec_loss')

        # obtain the residual vector of linear dynamics

        residual_vector_lin_loss = self.kphi - self.fdphidx
        residual_vector_lin_loss = tf.identity(residual_vector_lin_loss, name='residual_lin_loss')

        return residual_vector_rec_loss, residual_vector_lin_loss

    def get_graph(self):
        return self.graph

    def compile_optimization(self):
        """Choose optimizer and get summary done."""

        # self.optimizer = L4.L4Adam(fraction=0.2) # L4 optimizer, seems to be automatic step-size selection.... https://github.com/martius-lab/l4-optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_params['learningRate'],
                                                epsilon=self.model_params['decay'])

        # define minimize loss opt

        self.train_op = self.optimizer.minimize(self.loss_op)

        # use tensorflow summary to record everything
        tf.summary.scalar('total_loss', self.loss_op)
        tf.summary.scalar('rec_loss', self.rec_loss)
        tf.summary.scalar('norm_rec_loss(train)', self.norm_rec_loss)
        tf.summary.scalar('lin_loss', self.lin_loss)
        tf.summary.scalar('reg_loss', self.reg_parameter_loss)

        # merge all summary as a operation"""  # self.merged_summary_op = tf.summary.merge_all()  # self.train_writer
        # = tf.summary.FileWriter(self.dir + '/tbTrain', self.sess.graph)  # self.valid_writer = tf.summary.FileWriter(self.dir + '/tbValid')

    def buildSpecialMLPWeightBias(self, structure_list, prefix):

        weights = {}
        biases = {}

        # Setup the number of layers.
        # note that last layer doesn't contain bias
        numWeightedLayers = len(structure_list) - 1

        for indexLayer in xrange(numWeightedLayers):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               # Setup the shape of weights and biases for each layer.
            # naming each weight sand biases with `w` and `b` in the middle.
            weights_name = prefix + '_w_' + str(indexLayer + 1)
            weightsShape = (structure_list[indexLayer], structure_list[indexLayer + 1])

            if self.model_params['SVD_DMD']:
                if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                    biasShape = (1, structure_list[indexLayer + 1])
                    bias_name = prefix + '_b_' + str(indexLayer + 1)
            else:
                biasShape = (1, structure_list[indexLayer + 1])
                bias_name = prefix + '_b_' + str(indexLayer + 1)

            if self.model_params['enable_edward']:

                if self.model_params['edward_cfg']['mode'] == 'MAPSimple':

                    # building prior for weights and biases
                    # -- note that, in simple MAP, we abandon hyperprior
                    # -- note that the weights are indexed from 1

                    weights[weights_name] = Normal(loc=tf.zeros(weightsShape, dtype=TF_FLOAT),
                                                   scale=tf.ones(weightsShape, dtype=TF_FLOAT), name=weights_name)

                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                                       scale=tf.ones(biasShape, dtype=TF_FLOAT), name=bias_name)
                    else:
                        biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                                   scale=tf.ones(biasShape, dtype=TF_FLOAT), name=bias_name)

                    # adding the priors into dictionary

                    self.prior_dict[weights_name] = weights[weights_name]
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.prior_dict[bias_name] = biases[bias_name]
                    else:
                        self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = PointMass(
                        params=tf.get_variable(weights_name + "/loc", weightsShape, initializer=self.initializer,
                                               dtype=TF_FLOAT))
                    # -- biases

                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.vp_dict[bias_name] = PointMass(
                                params=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                                       dtype=TF_FLOAT))
                    else:
                        self.vp_dict[bias_name] = PointMass(
                            params=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                                   dtype=TF_FLOAT))

                elif self.model_params['edward_cfg']['mode'] == 'ADVInoARD':

                    # building prior for weights and biases

                    weights[weights_name] = self._create_sm2g_prior(list(weightsShape), weights_name)
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            biases[bias_name] = self._create_sm2g_prior(list(biasShape), bias_name)
                    else:
                        biases[bias_name] = self._create_sm2g_prior(list(biasShape), bias_name)

                    # adding the priors into dictionary
                    self.prior_dict[weights_name] = weights[weights_name]
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.prior_dict[bias_name] = biases[bias_name]
                    else:
                        self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = self._create_standard_normal(list(weightsShape), weights_name + "/loc",
                                                                              weights_name + "/scale", weights_name)
                    # -- biases
                    if self.model_params['SVD_DMD']:

                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.vp_dict[bias_name] = self._create_standard_normal(list(biasShape), bias_name + "/loc",
                                                                                   bias_name + "/scale", bias_name)
                    else:
                        self.vp_dict[bias_name] = self._create_standard_normal(list(biasShape), bias_name + "/loc",
                                                                               bias_name + "/scale", bias_name)

                elif self.model_params['edward_cfg']['mode'] == 'ADVIARD':

                    # scale parameter for weights and biases

                    self.hpp_dict[weights_name] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(weightsShape, dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(weightsShape, dtype=TF_FLOAT))

                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.hpp_dict[bias_name] = self._create_half_cauchy_prior(
                                [self.ALPHA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT),
                                [self.BETA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT))
                    else:
                        self.hpp_dict[bias_name] = self._create_half_cauchy_prior(
                            [self.ALPHA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT),
                            [self.BETA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT))

                    # self.hpp_dict[weights_name] = InverseGamma(
                    #     [self.ALPHA_HPP] * tf.ones(weightsShape),
                    #     [self.BETA_HPP] * tf.ones(weightsShape),
                    # name='scale' + weights_name)
                    #
                    # self.hpp_dict[bias_name] = InverseGamma(
                    #     [self.ALPHA_HPP] * tf.ones(biasShape),
                    #     [self.BETA_HPP] * tf.ones(biasShape),
                    # name='scale' + bias_name)

                    # building prior for weights and biases

                    weights[weights_name] = Normal(loc=tf.zeros(weightsShape, dtype=TF_FLOAT),
                                                   scale=tf.sqrt(self.hpp_dict[weights_name]), name=weights_name)
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                                       scale=tf.sqrt(self.hpp_dict[bias_name]), name=bias_name)
                    else:
                        biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                                   scale=tf.sqrt(self.hpp_dict[bias_name]), name=bias_name)

                    # adding the priors into dictionary

                    self.prior_dict[weights_name] = weights[weights_name]
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.prior_dict[bias_name] = biases[bias_name]
                    else:
                        self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = NormalWithSoftplusScale(
                        loc=tf.get_variable(weights_name + "/loc", weightsShape, initializer=self.initializer,
                                            dtype=TF_FLOAT),
                        scale=tf.get_variable(weights_name + "/scale", weightsShape, initializer=self.scale_initializer,
                                              dtype=TF_FLOAT), name='q' + weights_name)
                    # -- biases
                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.vp_dict[bias_name] = NormalWithSoftplusScale(
                                loc=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                                    dtype=TF_FLOAT),
                                scale=tf.get_variable(bias_name + "/scale", biasShape, initializer=self.scale_initializer,
                                                      dtype=TF_FLOAT), name='q' + bias_name)
                    else:
                        self.vp_dict[bias_name] = NormalWithSoftplusScale(
                            loc=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                                dtype=TF_FLOAT),
                            scale=tf.get_variable(bias_name + "/scale", biasShape, initializer=self.scale_initializer,
                                                  dtype=TF_FLOAT), name='q' + bias_name)

                    # building the corresponding variational posterior for the variance in the weights and biases
                    # -- variance in weights

                    # define the variational posteriori for the variance of weights
                    self.vp_dict['scale_' + weights_name] = self._create_scale_log_normal(weightsShape,
                        'qscale_' + weights_name + "/loc", 'qscale_' + weights_name + "/scale", 'qscale' + weights_name)

                    if self.model_params['SVD_DMD']:
                        if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                            self.vp_dict['scale_' + bias_name] = self._create_scale_log_normal(biasShape,
                                                                                               'qscale_' + bias_name + "/loc",
                                                                                               'qscale_' + bias_name + "/scale",
                                                                                               'qscale' + bias_name)
                    else:
                        self.vp_dict['scale_' + bias_name] = self._create_scale_log_normal(biasShape,
                                                                                           'qscale_' + bias_name + "/loc",
                                                                                           'qscale_' + bias_name + "/scale",
                                                                                           'qscale' + bias_name)

                    # self.vp_dict['scale_' + weights_name] = \  #     self.create_scale_log_normal(weightsShape, "q" + 'scale_' + weights_name + "/loc",  #                              "q" + 'scale_' + weights_name + "/scale", 'qscale' + weights_name)

                    # -- variance in biases  # """define the variational posteriori for the variance of biases"""  # self.vp_dict['scale_' + bias_name] = \  #     Normal(loc=tf.get_variable("q" + 'scale_' + bias_name + "/loc",  #                                biasShape, initializer=self.initializer),  #            scale=tf.nn.softplus(tf.get_variable("q" + 'scale_' + bias_name + "/scale",  #                                                 biasShape, initializer=self.scale_initializer)),  #            name='qscale' + bias_name)

                    # self.vp_dict['scale_' + bias_name] = \  #     self.create_scale_log_normal(biasShape, "q" + 'scale_' + bias_name + "/loc",  #                                  "q" + 'scale_' + bias_name + "/scale", 'qscale' + bias_name)

            else:

                # if edward is not used, simply use `tf.Variable` to build weights and biases.

                weights[prefix + '_w_' + str(indexLayer + 1)] = tf.Variable(self.initializer(weightsShape),
                    dtype=TF_FLOAT)
                if self.model_params['SVD_DMD']:
                    if (prefix == 'enc' and indexLayer < numWeightedLayers - 1) or (prefix == 'dec' and indexLayer > 0):
                        biases[prefix + '_b_' + str(indexLayer + 1)] = tf.Variable(self.initializer(biasShape), dtype=TF_FLOAT)
                else:
                    biases[prefix + '_b_' + str(indexLayer + 1)] = tf.Variable(self.initializer(biasShape), dtype=TF_FLOAT)

        # record the number of layers

        self.hpp_dict[prefix + '_number_layer'] = numWeightedLayers

        return weights, biases

    def buildMLPWeightsBias(self, structure_list, prefix):
        """build FNN weights and biases given `structure_list`.

        Args:
            structure_list (:obj:`list`): the list describing the structure of FNN encoder.

            prefix (:obj:`str`): `enc` or `dec`.

        Returns:

            :obj:`tuple`: (:obj:`dict`, :obj:`dict`) as collection of weights and biases.

        """

        weights = {}
        biases = {}

        # Setup the number of layers.

        numWeightedLayers = len(structure_list) - 1

        for indexLayer in xrange(numWeightedLayers):

            # Setup the shape of weights and biases for each layuer.

            weightsShape = (structure_list[indexLayer], structure_list[indexLayer + 1])
            biasShape = (1, structure_list[indexLayer + 1])

            # naming each weight sand biases with `w` and `b` in the middle.

            weights_name = prefix + '_w_' + str(indexLayer + 1)
            bias_name = prefix + '_b_' + str(indexLayer + 1)

            if self.model_params['enable_edward']:

                if self.model_params['edward_cfg']['mode'] == 'MAPSimple':

                    # building prior for weights and biases
                    # -- note that, in simple MAP, we abandon hyperprior
                    # -- note that the weights are indexed from 1

                    weights[weights_name] = Normal(loc=tf.zeros(weightsShape, dtype=TF_FLOAT),
                                                   scale=tf.ones(weightsShape, dtype=TF_FLOAT), name=weights_name)
                    biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                               scale=tf.ones(biasShape, dtype=TF_FLOAT), name=bias_name)

                    # adding the priors into dictionary

                    self.prior_dict[weights_name] = weights[weights_name]
                    self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = PointMass(
                        params=tf.get_variable(weights_name + "/loc", weightsShape, initializer=self.initializer,
                                               dtype=TF_FLOAT))
                    # -- biases

                    self.vp_dict[bias_name] = PointMass(
                        params=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                               dtype=TF_FLOAT))

                elif self.model_params['edward_cfg']['mode'] == 'ADVInoARD':

                    # building prior for weights and biases

                    weights[weights_name] = self._create_sm2g_prior(list(weightsShape), weights_name)
                    biases[bias_name] = self._create_sm2g_prior(list(biasShape), bias_name)

                    # adding the priors into dictionary

                    self.prior_dict[weights_name] = weights[weights_name]
                    self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = self._create_standard_normal(list(weightsShape), weights_name + "/loc",
                                                                              weights_name + "/scale", weights_name)
                    # -- biases

                    self.vp_dict[bias_name] = self._create_standard_normal(list(biasShape), bias_name + "/loc",
                                                                           bias_name + "/scale", bias_name)

                elif self.model_params['edward_cfg']['mode'] == 'ADVIARD':

                    # scale parameter for weights and biases

                    self.hpp_dict[weights_name] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(weightsShape, dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(weightsShape, dtype=TF_FLOAT))

                    self.hpp_dict[bias_name] = self._create_half_cauchy_prior(
                        [self.ALPHA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT),
                        [self.BETA_HPP] * tf.ones(biasShape, dtype=TF_FLOAT))

                    # self.hpp_dict[weights_name] = InverseGamma(
                    #     [self.ALPHA_HPP] * tf.ones(weightsShape),
                    #     [self.BETA_HPP] * tf.ones(weightsShape),
                    # name='scale' + weights_name)
                    #
                    # self.hpp_dict[bias_name] = InverseGamma(
                    #     [self.ALPHA_HPP] * tf.ones(biasShape),
                    #     [self.BETA_HPP] * tf.ones(biasShape),
                    # name='scale' + bias_name)

                    # building prior for weights and biases

                    weights[weights_name] = Normal(loc=tf.zeros(weightsShape, dtype=TF_FLOAT),
                                                   scale=tf.sqrt(self.hpp_dict[weights_name]), name=weights_name)
                    biases[bias_name] = Normal(loc=tf.zeros(biasShape, dtype=TF_FLOAT),
                                               scale=tf.sqrt(self.hpp_dict[bias_name]), name=bias_name)

                    # adding the priors into dictionary

                    self.prior_dict[weights_name] = weights[weights_name]
                    self.prior_dict[bias_name] = biases[bias_name]

                    # building variatioal posterior for weights and biases
                    # -- weights

                    self.vp_dict[weights_name] = NormalWithSoftplusScale(
                        loc=tf.get_variable(weights_name + "/loc", weightsShape, initializer=self.initializer,
                                            dtype=TF_FLOAT),
                        scale=tf.get_variable(weights_name + "/scale", weightsShape, initializer=self.scale_initializer,
                                              dtype=TF_FLOAT), name='q' + weights_name)
                    # -- biases

                    self.vp_dict[bias_name] = NormalWithSoftplusScale(
                        loc=tf.get_variable(bias_name + "/loc", biasShape, initializer=self.initializer,
                                            dtype=TF_FLOAT),
                        scale=tf.get_variable(bias_name + "/scale", biasShape, initializer=self.scale_initializer,
                                              dtype=TF_FLOAT), name='q' + bias_name)

                    # building the corresponding variational posterior for the variance in the weights and biases
                    # -- variance in weights

                    # define the variational posteriori for the variance of weights
                    self.vp_dict['scale_' + weights_name] = self._create_scale_log_normal(weightsShape,
                        'qscale_' + weights_name + "/loc", 'qscale_' + weights_name + "/scale", 'qscale' + weights_name)
                    self.vp_dict['scale_' + bias_name] = self._create_scale_log_normal(biasShape,
                                                                                       'qscale_' + bias_name + "/loc",
                                                                                       'qscale_' + bias_name + "/scale",
                                                                                       'qscale' + bias_name)

                    # self.vp_dict['scale_' + weights_name] = \  #     self.create_scale_log_normal(weightsShape, "q" + 'scale_' + weights_name + "/loc",  #                              "q" + 'scale_' + weights_name + "/scale", 'qscale' + weights_name)

                    # -- variance in biases  # """define the variational posteriori for the variance of biases"""  # self.vp_dict['scale_' + bias_name] = \  #     Normal(loc=tf.get_variable("q" + 'scale_' + bias_name + "/loc",  #                                biasShape, initializer=self.initializer),  #            scale=tf.nn.softplus(tf.get_variable("q" + 'scale_' + bias_name + "/scale",  #                                                 biasShape, initializer=self.scale_initializer)),  #            name='qscale' + bias_name)

                    # self.vp_dict['scale_' + bias_name] = \  #     self.create_scale_log_normal(biasShape, "q" + 'scale_' + bias_name + "/loc",  #                                  "q" + 'scale_' + bias_name + "/scale", 'qscale' + bias_name)

            else:

                # if edward is not used, simply use `tf.Variable` to build weights and biases.

                weights[prefix + '_w_' + str(indexLayer + 1)] = tf.Variable(self.initializer(weightsShape),
                    dtype=TF_FLOAT)
                biases[prefix + '_b_' + str(indexLayer + 1)] = tf.Variable(self.initializer(biasShape), dtype=TF_FLOAT)

        # record the number of layers

        self.hpp_dict[prefix + '_number_layer'] = numWeightedLayers

        return weights, biases

    def buildPODWeights(self, structure_list, prefix):
        """Build POD short cuts weights for the encoder and decoder.

        Essentially it is just doing SVD on the input data.

        Args:
            structure_list (:obj:`list`): structure of the FNN encoder.

            prefix (:obj:`str`): striong of the prefix.

        Returns:
            :obj:`dict` : dictionary of POD weights.

        """

        weights = {}
        # weightsShape = (structure_list[0], structure_list[-1])

        # enable pod? if so, we fix the weight as constant

        print '=============================================='
        print 'Compute FIXED POD weights for the SHORT CUTS!'
        print '=============================================='

        # init the weights using POD

        if prefix == 'enc':

            shape_PODV = self.POD_V.get_shape().as_list()
            shape_PODV_col = shape_PODV[1]
            num_Koopmanmodes = structure_list[-1]

            # print num_Koopmanmodes, shape_PODV

            if shape_PODV_col < num_Koopmanmodes:

                # if mapping to a larger dimension, we zero padding the rest
                # zero padding for POD

                zero_column = tf.zeros((shape_PODV_col, num_Koopmanmodes - shape_PODV_col), dtype=TF_FLOAT)
                self.POD_V = tf.concat([self.POD_V, zero_column], axis=1)
                weights[prefix + '_POD'] = self.POD_V  # tf.convert_to_tensor(self.POD_V, dtype=tf.float32)

            else:

                # if mapping to a smaller or equal dimension, we only map the first few POD coordinate over there

                weights[prefix + '_POD'] = self.POD_V[:, :num_Koopmanmodes]

            print 'encoder part: POD weight = \n', self.sess.run(weights[prefix + '_POD'])
            print 'encoder part: POD weight shape: \n', self.sess.run(weights[prefix + '_POD']).shape

        else:

            # decoder case

            shape_PODV = self.POD_V.get_shape().as_list()
            shape_PODV_col = shape_PODV[1]
            num_Koopmanmodes = structure_list[0]

            # print num_Koopmanmodes, shape_PODV

            # if mapping to a larger dimension, we zero padding the rest

            if shape_PODV_col < num_Koopmanmodes:

                # zero padding for POD

                zero_column = tf.zeros((shape_PODV_col, num_Koopmanmodes - shape_PODV_col), dtype=TF_FLOAT)
                self.POD_V = tf.concat([self.POD_V, zero_column], axis=1)
                weights[prefix + '_POD'] = self.POD_V  # tf.convert_to_tensor(self.POD_V, dtype=tf.float32)

            else:

                weights[prefix + '_POD'] = self.POD_V[:, :num_Koopmanmodes]

            # we just do a extra transpose, we get the decoder POD
            weights[prefix + '_POD'] = tf.transpose(weights[prefix + '_POD'])

            print 'decoder part: POD weight = \n', self.sess.run(weights[prefix + '_POD'])
            print 'decoder part: POD weight shape: \n', self.sess.run(weights[prefix + '_POD']).shape

        return weights

    def createMLPOutLinearOLD(self, start_layer, weights, bias, prefix):
        """FNN layer with ``start_layer`` as input.

        It takes the ``phi_dmd`` into consideration about whether or not add SVD_DMD into it.

        Args:
            start_layer (:obj:`tf.Tesnor`): input tensor.

            weights (:obj:`dict`): collection of weights.

            bias (:obj:`dict`): collection of biases.

            prefix (:obj:`str`): prefix to the layer.

        Returns:
            :obj:`tuple` : last output of the feedforward, and last output of the feedforward without activations in
                between.

        """

        MLP = [start_layer]
        numWeightedLayers = len(weights)

        # if prefix == 'enc':
        #
        #     start_index = 1
        #     end_index = numWeightedLayers
        #     offset_of_initial_layer = -1
        #
        # elif prefix == 'dec':
        #
        #     start_index = 2
        #     end_index = numWeightedLayers + 1
        #     offset_of_initial_layer = -2
        #
        # else:
        #     raise NotImplementedError('not implemented! check prefix!')

        # by doing this, we cna have the desired FNN we want, with desired weights and biases
        for indexLayer in xrange(1, numWeightedLayers + 1):

            before_act = tf.add(tf.matmul(MLP[indexLayer - 1],
                                          weights[prefix + '_w_' + str(indexLayer)]),
                                bias[prefix + '_b_' + str(indexLayer)])

            if indexLayer < numWeightedLayers:
                output = self.activation(before_act)
            else:
                output = before_act

            MLP.append(output)

        return MLP[-1]

    def createMLPOutLinear(self, start_layer, weights, bias, prefix):
        """FNN layer with ``start_layer`` as input.

        It takes the ``phi_dmd`` into consideration about whether or not add SVD_DMD into it.

        Args:
            start_layer (:obj:`tf.Tesnor`): input tensor.

            weights (:obj:`dict`): collection of weights.

            bias (:obj:`dict`): collection of biases.

            prefix (:obj:`str`): prefix to the layer.

        Returns:
            :obj:`tuple` : last output of the feedforward, and last output of the feedforward without activations in
                between.

        """

        MLP = [start_layer]
        numWeightedLayers = len(weights)

        if prefix == 'enc':

            start_index = 1
            end_index = numWeightedLayers
            offset_of_initial_layer = -1

            # by doing this, we can have the desired FNN we want, with desired weights and biases
            for indexLayer in xrange(start_index, end_index):

                before_act = tf.add(tf.matmul(MLP[indexLayer + offset_of_initial_layer],
                                              weights[prefix + '_w_' + str(indexLayer)]),
                                    bias[prefix + '_b_' + str(indexLayer)])

                if indexLayer < end_index - 1:
                    output = self.activation(before_act)
                else:
                    output = before_act

                MLP.append(output)


        elif prefix == 'dec':

            start_index = 2
            end_index = numWeightedLayers + 1
            offset_of_initial_layer = -2

            # by doing this, we can have the desired FNN we want, with desired weights and biases
            for indexLayer in xrange(start_index, end_index):

                before_act = tf.add(tf.matmul(MLP[indexLayer + offset_of_initial_layer],
                                              weights[prefix + '_w_' + str(indexLayer)]),
                                    bias[prefix + '_b_' + str(indexLayer)])

                if indexLayer < end_index - 1:
                    output = self.activation(before_act)
                else:
                    output = before_act

                MLP.append(output)

        else:
            raise NotImplementedError('not implemented! check prefix!')


        return MLP[-1]

    def initilize(self):
        """initialize session and variables"""

        # initalize all variables
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def getX_Xdot(self, X, Xdot, valid_size=0.01):
        """get training data & validation data with a certain valid_size split

        Note:
            :attr:`model_params` contains ``normalization_type``, which will indicates
            how we treat the normalization process. ``only_max`` means we will keep the ratio of variance.
            Otherwise, we just do treat everyone into standard deviation 1.

        Args:
            X (:obj:`numpy.ndarray`): training data states :math:`X`, with ``axis=0`` corresponding to number of data
                snapshots.

            Xdot (:obj:`numpy.ndarray`) training data time derivative :math:`\dot{x}`.

            valid_size (:obj:`float`): validation split ratio.

        """

        # read unnormalized data and split into train and test

        X_train, X_valid, Xdot_train, Xdot_valid, _, _ = \
            train_test_split(X, Xdot, test_size=valid_size, random_state=19901012)

        self.Xtrain = X_train
        self.XdotTrain = Xdot_train
        self.Xvalid = X_valid
        self.XdotValid = Xdot_valid

        # normalizing X
        scaler_X = StandardScaler()
        scaler_X.fit(X_train)

        if self.model_params['normalization_type'] == 'only_max':
            variance = np.ones(scaler_X.var_.shape) * np.max(scaler_X.var_)
            self.mu_X = tf.convert_to_tensor(scaler_X.mean_, dtype=TF_FLOAT, name='mu_X')
            self.Lambda_X = tf.convert_to_tensor(np.diag(np.sqrt(variance)), dtype=TF_FLOAT, name='Lambda_X')
            self.Lambda_X_inv = tf.linalg.inv(self.Lambda_X)

            # create self.POD_V based on eta_xtrain
            X_train_normalized = scaler_X.transform(self.Xtrain)
            u, s, vh = np.linalg.svd(X_train_normalized, full_matrices=False)

        elif self.model_params['normalization_type'] == 'not_only_max':
            self.mu_X = tf.convert_to_tensor(scaler_X.mean_, dtype=TF_FLOAT, name='mu_X')
            variance = scaler_X.var_
            self.Lambda_X = tf.convert_to_tensor(np.diag(np.sqrt(variance)), dtype=TF_FLOAT, name='Lambda_X')
            self.Lambda_X_inv = tf.linalg.inv(self.Lambda_X)

            # create self.POD_V based on eta_xtrain
            X_train_normalized = scaler_X.transform(self.Xtrain)
            u, s, vh = np.linalg.svd(X_train_normalized, full_matrices=False)
        else:
            variance = scaler_X.var_
            self.mu_X = tf.convert_to_tensor(scaler_X.mean_*0, dtype=TF_FLOAT, name='mu_X')
            self.Lambda_X = self.Lambda_X = tf.convert_to_tensor(np.diag(np.ones(variance.size)), dtype=TF_FLOAT, name='Lambda_X')
            self.Lambda_X_inv = tf.linalg.inv(self.Lambda_X)

            # create self.POD_V based on eta_xtrain
            X_train_normalized = self.Xtrain
            u, s, vh = np.linalg.svd(X_train_normalized, full_matrices=False)

        # POD is obtained in eta space
        self.POD_V = tf.convert_to_tensor(vh.transpose(), dtype=TF_FLOAT)

    def shuffle_data_index(self, X):
        """Shuffling the data ``X`` with randomness.

        Args:
            X (:obj:`numpy.ndarray`): input data.

        Returns:
            :obj:`numpy.ndarray` : the index of random shuffle.
        """

        num_size = X.shape[0]
        random_index = np.random.permutation(num_size)
        return random_index

    def shuffle_data(self, X, Y):
        """shuffle data given X and Y

        Args:
            X (:obj:`numpy.ndarray`): X data
            Y (:obj:`numpy.darray`): Y data

        Returns:
            :obj:`tuple` : (random shuffled X, random shuffled Y).

        """

        num_size = X.shape[0]
        random_index = np.random.permutation(num_size)
        return X[random_index], Y[random_index]

    def train(self):
        """training the neural network with mini-batch training

        Note:
            :attr:`model_params` contains ``numberEpoch`` and ``miniBatch`` will lead to
            the use of number of epochs and batch size.

        """

        # randomize data before training
        self.Xtrain, self.XdotTrain = self.shuffle_data(X=self.Xtrain, Y=self.XdotTrain)

        # get the number of epoches and batch size
        N_EPOCHS = self.model_params['numberEpoch']
        BATCH_SIZE = self.model_params['miniBatch']

        train_count = self.Xtrain.shape[0]

        # recording loss
        self.loss_dict = {}
        self.loss_dict['total_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['linear_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['recon_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['reg_MSE'] = {'train': [], 'valid': []}

        for i in range(1, N_EPOCHS + 1):

            # mini-batch training
            for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                # training on a subsets of training set
                _ = self.sess.run([self.train_op],
                                  feed_dict={self.X: self.Xtrain[start:end], self.Xdot: self.XdotTrain[start:end]})

            # reocrding in high frequency for plotting
            total_loss_train, linear_loss_train, recon_loss_train, reg_loss_train = self.sess.run(
                [self.loss_op, self.lin_loss, self.rec_loss, self.reg_parameter_loss],
                feed_dict={self.X: self.Xtrain, self.Xdot: self.XdotTrain})
            total_loss_valid, linear_loss_valid, recon_loss_valid, reg_loss_valid = self.sess.run(
                [self.loss_op, self.lin_loss, self.rec_loss, self.reg_parameter_loss],
                feed_dict={self.X: self.Xvalid, self.Xdot: self.XdotValid})

            self.loss_dict['total_MSE']['train'].append(total_loss_train)
            self.loss_dict['total_MSE']['valid'].append(total_loss_valid)
            self.loss_dict['linear_MSE']['train'].append(linear_loss_train)
            self.loss_dict['linear_MSE']['valid'].append(linear_loss_valid)
            self.loss_dict['recon_MSE']['train'].append(recon_loss_train)
            self.loss_dict['recon_MSE']['valid'].append(recon_loss_valid)
            self.loss_dict['reg_MSE']['train'].append(reg_loss_train)
            self.loss_dict['reg_MSE']['valid'].append(reg_loss_valid)

            # recording in low frequency for tensorboard and for printing
            if i % 100 == 0:
                ## evaluating whole training data sets
                # record metadata in running
                train_loss, K = self.sess.run([self.loss_op, self.koopmanOp_learn],
                                              feed_dict={self.X: self.Xtrain, self.Xdot: self.XdotTrain})

                # evaluate the whole validation loss
                valid_loss = self.sess.run([self.loss_op], feed_dict={self.X: self.Xvalid, self.Xdot: self.XdotValid})

                print '=============='
                print 'epoch = ', i
                print 'train_loss = ', train_loss
                print 'valid_loss = ', valid_loss
                print 'koopman operator = \n', K
                print 'koopman eigenvalue = \n', LA.eig(K)[0]
                print ''


    def Save_pps_data_to_disk(self, total_MSE_list_from_edward=None):

        """Save data for plotting the learning curves.

        Note:

            - plot learning curve for train and validation
            - plot a priori scattering for train and validation

        Args:
            total_MSE_list_from_edward (:obj:`list`): if bayesian is enabled, it will feed the loss
                function from edward to here.

        """

        # Koopman eigenstuff
        self.SaveComputeKoopman()

        # Note
        # -- if edward is enabled, I would manually add the loss into it
        # -- but I found it is difficult to separate linear loss and reconstruction loss
        # -- so I will stick with total loss, call it a MSE

        if self.model_params['enable_edward']:

            self.loss_dict = {}
            self.loss_dict['total_MSE'] = {'train': total_MSE_list_from_edward['train'],
                                           'valid': total_MSE_list_from_edward['valid']}

            # plot learning curve
            self.SaveLearningCurve(prefix='total-MSE', train_metrics_list=self.loss_dict['total_MSE']['train'],
                                   valid_metrics_list=self.loss_dict['total_MSE']['valid'])
        else:

            # plot learning curve
            self.SaveLearningCurve(prefix='total-MSE', train_metrics_list=self.loss_dict['total_MSE']['train'],
                                   valid_metrics_list=self.loss_dict['total_MSE']['valid'])
            self.SaveLearningCurve(prefix='linear-MSE', train_metrics_list=self.loss_dict['linear_MSE']['train'],
                                   valid_metrics_list=self.loss_dict['linear_MSE']['valid'])
            self.SaveLearningCurve(prefix='recon-MSE', train_metrics_list=self.loss_dict['recon_MSE']['train'],
                                   valid_metrics_list=self.loss_dict['recon_MSE']['valid'])
            self.SaveLearningCurve(prefix='reg-MSE', train_metrics_list=self.loss_dict['reg_MSE']['train'],
                                   valid_metrics_list=self.loss_dict['reg_MSE']['valid'])

    def sample_K(self):
        """Sampling K matrix for Bayesian case

        Default is using 100 samples for Monte Carlo sampling.

        Note that stabilization is implied.

        """

        if self.model_params['nsd'] == 1:

            X_upper = self.sess.run(self.vp_dict['K_K_X'].sample(self.N_SAMPLES))
            K_SD = self.sess.run(self.vp_dict['K_SD'].sample(self.N_SAMPLES))

            K_list = []
            for i in xrange(self.N_SAMPLES):
                X_upper_k_1 = np.diag(np.diagonal(X_upper[i], offset=1), k=1)
                K = X_upper_k_1 - X_upper_k_1.T - np.diag(K_SD[i])
                K_list.append(K)

        else:

            K = self.sess.run(self.vp_dict['K'].sample(self.N_SAMPLES))

            K_list = []
            for i in xrange(self.N_SAMPLES):
                K_list.append(K[i])

        self.K_sample = np.array(K_list)

    def SaveComputeKoopman(self):
        """Save koopman related things: eigenvalues + eigenfunction (only when system is 2D)

        Note:

            - consider :math:`KR=RL` as :math:`L` eigenvalues and :math:`R` right eigenvectors,
                then :math:`d\phi/dt = \phi K \iff d(\phi R)/dt = (\phi R) L`.

            - The name of the file for eigvals will be ``koopman_eigenvalue_deeplearn.npz``

        """

        with self.graph.as_default():

            if self.model_params['enable_edward']:

                # 1. compute Koopman eigenvalues

                self.sample_K()

                # -- sample Koopman K realization
                koopmanOp_learn_realization = self.K_sample

                D_real_list = []
                D_imag_list = []
                R_list = []

                for i in xrange(self.N_SAMPLES):
                    kmatrix = koopmanOp_learn_realization[i]
                    [D, R] = LA.eig(kmatrix)
                    D_real = np.real(D)
                    D_imag = np.imag(D)
                    D_real_list.append(D_real)
                    D_imag_list.append(D_imag)
                    R_list.append(R)

                print 'eigen system size: ', D_real.shape
                # save samples of Koopman eigenvalues
                np.savez(self.dir + '/koopman_eigenvalue_deeplearn.npz',
                         eig_real=D_real_list, eig_imag=D_imag_list)

            else:

                # 1. compute Koopman eigenvalues
                kmatrix = self.sess.run(self.koopmanOp_learn)
                [D, R] = LA.eig(kmatrix)
                D_real = np.real(D)
                D_imag = np.imag(D)

                # 1++ save Koopman eigenvalues data
                np.savez(self.dir + '/koopman_eigenvalue_deeplearn.npz', eig_real=D_real, eig_imag=D_imag)

                # 2. visualize eigenfunction when the physical space dimension is 2D
                if self.model_params['structureEncoder'][0] == 2:
                    # number of koopman modes
                    numKoopmanModes = self.model_params['structureEncoder'][-1]

                    # distribute sampling point in physical space
                    x1_min, x1_max = self.model_params['phase_space_range'][0]
                    x2_min, x2_max = self.model_params['phase_space_range'][1]

                    ndraw = 100
                    ndrawj = ndraw * 1j
                    x1_, x2_ = np.mgrid[x1_min:x1_max:ndrawj, x2_min:x2_max:ndrawj]

                    sample_x = x1_.reshape(-1, 1)
                    sample_xdot = x2_.reshape(-1, 1)

                    x_sample = np.hstack((sample_x, sample_xdot))  # make it (?,2) size

                    assert x_sample.shape == (
                        ndraw ** 2, 2), "sampling shape is wrong, check visualizing eigenfunction!"

                    # compute Koopman eigenfunction
                    phi_array = self.sess.run(self.koopmanPhi,
                                              feed_dict={self.X: x_sample, self.Xdot: np.zeros(x_sample.shape)})

                    ## SAVE
                    np.savez(self.dir + '/koopman_eigenfunctions.npz', numKoopmanModes=numKoopmanModes, R=R,
                             phi_array=phi_array, ndraw=ndraw, D_real=D_real, D_imag=D_imag, x1_=x1_, x2_=x2_)

    def SaveLearningCurve(self, prefix, train_metrics_list, valid_metrics_list):
        """Save training and validation error metrics vs training epoch

        Note:
            File will be saved as ``*_learn_curve.npz``.

        Args:
            prefix (:obj:`str`): prefix to the saving learning curve case.

            train_metrics_list (:obj:`list`): training loss list.

            valid_metrics_list (:obj:`list`): validation loss list.

        """

        np.savez(self.dir + '/' + prefix + '_learn_curve.npz', train_metrics_list=train_metrics_list,
                 valid_metrics_list=valid_metrics_list)

    def summary(self):
        """ Print number of all trainable weights and data points

        """

        print '==== Summary ====='
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print 'number of parameters to train: ', total_parameters
        print 'number of training data points total dim: ', self.Xtrain.size
        print 'number of validation data points total dim: ', self.Xvalid.size

    def Save_train_data(self, data):
        """ Save training data points distribution in phase space

        Note:
            it will be saved as ``train_data.npz``.

        Args:
            data (:obj:`numpy.ndarray`): data snapshots.

        """
        np.savez(self.dir + '/train_data.npz', data=data)

    def save_parameter_npy(self):
        """Save neural network encoder decoder parameters.

        Note:

            It will be named as ``saved_para.npy``.

            - save encoder weights, bias
            - save decoder weights, bias
            - save K

        """

        total_para = {}
        weight_bias_K_save = {}

        NUM_MC_SAMPLES = 500

        if self.model_params['enable_edward']:

            # if saving weights of Bayes, you shouldn't save prior weights,
            # but to save vp_dict results

            # ``Bayesian-saving weight and biases``

            # search for all the weight and biases in vp_dict
            for key in self.vp_dict:
                if bool(re.match("enc_[a-z]_[0-9]", key)) or bool(re.match("dec_[a-z]_[0-9]", key)):
                    print('saving weights and biases = ', key)
                    # start to sample weight and biases
                    weight_bias_K_save[key] = self.sess.run(self.vp_dict[key].sample(NUM_MC_SAMPLES))

            # then we find K in vp_dict and we get it sampled
            if self.model_params['nsd'] == 1:

                X_upper = self.sess.run(self.vp_dict['K_K_X'].sample(NUM_MC_SAMPLES))
                K_SD = self.sess.run(self.vp_dict['K_SD'].sample(NUM_MC_SAMPLES))

                K_list = []
                for i in xrange(NUM_MC_SAMPLES):
                    X_upper_k_1 = np.diag(np.diagonal(X_upper[i], offset=1), k=1)
                    K = X_upper_k_1 - X_upper_k_1.T - np.diag(K_SD[i])
                    K_list.append(K)

                K_samples = np.array(K_list)

            else:

                K_samples = self.sess.run(self.vp_dict['K'].sample(NUM_MC_SAMPLES))

            weight_bias_K_save['K'] = K_samples


            # in the following total_para
            for key in self.vp_dict:
                try:
                    total_para[key] = self.sess.run(self.vp_dict[key].mean())
                except NotImplementedError:
                    print('not being able to sample the mean of ', key, ' so we compute sampled mean with 100 samples')
                    total_para[key] = np.mean(self.sess.run(self.vp_dict[key].sample(100)),axis=0)

        else:

            # deterministic model

            # -- 1. save encoder weights
            for key in self.encoderLayerWeights:
                total_para[key] = self.sess.run(self.encoderLayerWeights[key])
            # -- 2. save encoder Bias
            for key in self.encoderLayerBias:
                total_para[key] = self.sess.run(self.encoderLayerBias[key])
            # -- 3. save decoder Weights
            for key in self.decoderLayerWeights:
                total_para[key] = self.sess.run(self.decoderLayerWeights[key])
            # -- 4. save decoder Bias
            for key in self.decoderLayerBias:
                total_para[key] = self.sess.run(self.decoderLayerBias[key])

            # -- 5. save K
            total_para['K'] = self.sess.run(self.koopmanOp_learn)

        # save parameters
        np.save(self.dir + '/saved_para.npy', total_para)
        np.save(self.dir + '/weight_bias_save.npy', weight_bias_K_save)

    def save_model(self):
        """Save the whole model using ``tf.saved_model.simple_save``.

        Note:
            - create new folder as ``model_saved``
            - meta info is contained in ``model_arch.npz``.
        """
        print '=========================='
        print 'show the configuration'
        print self.model_params
        print '=========================='

        # save model
        with self.graph.as_default():
            model_dir = self.dir + '/model_saved'
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)


            # save model using Tensorflow simple save
            tf.saved_model.simple_save(self.sess, model_dir, inputs={"X": self.X, "Xdot": self.Xdot},
                                       outputs={"Xrec": self.Xrec})

            # save K, encoder, decoder parameter into numpy
            self.save_parameter_npy()

            # save arch data in model_arch.npz:
            # -- encoder layer structure
            # -- activation function
            # -- POD weights
            #   -- encoder_POD_weights
            #   -- decoder_POD_weights
            np.savez(model_dir + '/model_arch.npz',
                     encoder_layer_structure=self.model_params['structureEncoder'],
                     act_fun=self.model_params['activation'],
                     encoder_POD_weights=self.sess.run(self.encoderPODWeights['enc_POD']),
                     decoder_POD_weights=self.sess.run(self.decoderPODWeights['dec_POD']),
                     SVD_DMD=self.model_params['SVD_DMD'],
                     nonlinear_rec=self.model_params['typeRecon']=='nonlinear',
                     nsd=self.model_params['nsd'])


if __name__ == '__main__':
    pass
