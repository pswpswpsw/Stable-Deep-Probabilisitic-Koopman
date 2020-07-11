#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""It contains three classs of Continuous time Deep Learning Models for Koopman Operator with Bayes

    Class List:

    1. :class:`edwardMBTrainer`
    2. :class:`edwardMBTrainerLRAN`
    3. :class:`ClassBayesDLKoopman`

"""
import os
os.environ['TF_C_API_GRAPH_CONSTRUCTION']='0'
import sys
import numpy as np
import edward as ed
sys.dont_write_bytecode = True
import GPUtil
import tensorflow as tf

from ClassDLKoopmanCont import *
from ClassDLKoopmanLRAN import *
from lib.utilities import TF_FLOAT
from edward.models import InverseGamma, MultivariateNormalDiag, Cauchy
from tensorflow.contrib.distributions import bijectors

try:
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.1, maxMemory=0.1)
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

except:
    print('CPU mode')




class ClassBayesDLKoopman(object):
    """Class of Deep Leraning Koopman with Bayesian Deep Learning.

    Args:
        data (:obj:`numpy.ndarray`): training data with key ``Xtrain`` and ``XdotTrain`` for :attr:`LRAN_mode` =
            ``False`` while only with ``Xtrain`` otherwise.

        configDict (:obj:`dict`): config dictionary for deterministic DL Koopman

        mc_sample (:obj:`int`): number of Monte carlo sampling in using the reparameterization trick.
            We recommend 1. The higher it is, the less variance there is in the gradient estimation.

        edward_configDict (:obj:`dict`): config dictionary for bayesian DL Koopman.

        cut_data_range (:obj:`numpy.ndarray`): Specifiy the range for training data. If ``None``, then we don't use it.

        LRAN_mode (:obj:`bool`): Using LRAN or not.

        gpu_percentage (:obj:`float`): gpu memory usage ratio. Default value is 0.5.

    Attributes:

        model (:obj:`class`): Deterministic DL Koopman model class from ``ClassDLKoopmanCont``.

        residual_vector_rec_loss (:obj:`tf.Tensor`): residual reconstruction loss vector as the output of deterministic
            network as likelihood.

        residual_vector_lin_loss (:obj:`tf.Tensor`): residual linear loss vector as the output of deterministic
            network as likelihood.

        z_ph (:obj:`tf.Tensor`): concanteted ``z_rec_loss`` and ``z_lin_loss`` as general loss target to minimize.

        vp_dict (:obj:`dict`): dictionary collection of variational posteriori

        sess (:obj:`class`): current session used in Tensorflow.

        graph (:obj:`class`): current graph being used.

        verbose (:obj:`bool`): logging flags.

        mode (:obj:`str`): choose which mode to take on ADVI.

        LRAN_mode (:obj:`bool`): choose using ``recurrent`` or ``differential`` form.

        dir (:obj:`str`): path to saving directory.

        mc_sample (:obj:`int`): number of MC samples to choose for approximating the reparameterization trick.

        init_mode (:obj:`str`): ``Standard`` or ``nonBayes`` as using default weights initialization or simply use previous run.

        MAP_dir (:obj:`str`): if :attr:`init_mode` is turned with ``nonBayes``, then one needs to give the path to previous trained model.

        init_std_softplus (:obj:`float`): initialization of the standard deviation for softplus for the scales. The smaller the more negative, the weights will be smaller.

        optimizer (:obj:`str`): the name for the optimizer in tensorflow used in the training process. For example, ``adam``.
    """

    def __init__(self, data, configDict, mc_sample, edward_configDict, cut_data_range=None, LRAN_mode=None, gpu_percentage=0.5):

        # construct the graph in Cont model

        if LRAN_mode == None:

            ## Differential form Bayesian

            self.model = ClassDLKoopmanCont(configDict=configDict, edward_configDict=edward_configDict)

            # obtaining data

            if cut_data_range == None:

                self.model.getX_Xdot(X=data['Xtrain'], Xdot=data['XdotTrain'])

            else:

                x1_range = cut_data_range['x1_range']
                x2_range = cut_data_range['x2_range']

                index_after_cut = (data['Xtrain'][:, 0] <= x1_range[1]) & (data['Xtrain'][:, 0] >= x1_range[0]) & (
                            data['Xtrain'][:, 1] <= x2_range[1]) & (data['Xtrain'][:, 1] >= x2_range[0])

                print 'data size before cut = ', data['Xtrain'].shape[0]

                cut_Xtrain = data['Xtrain'][index_after_cut]
                cut_XdotTrain = data['XdotTrain'][index_after_cut]
                print 'data size after cut = ', cut_Xtrain.shape[0]

                self.model.getX_Xdot(X=cut_Xtrain, Xdot=cut_XdotTrain)

        else:

            ## LRAN form Bayesian
            self.model = ClassDLKoopmanLRAN(configDict=configDict, edward_configDict=edward_configDict, gpu_percentage=gpu_percentage)

            # only use Xtrain, not XdotTrain
            self.model.get_hankel_data(X=data['Xtrain'])


        # construct model graph with total reconstruction residual vector
        # and linear dynamical residual vector and prior dictionary
        # and hyperparameter dictionary defined
        # -- self.model.prior_dict: RVs for weights, biases and Koopman K
        # -- self.model.hpp_dict: RVs for the variance of the above RVs
        self.residual_vector_rec_loss, self.residual_vector_lin_loss = self.model.construct_model()

        # passing the auxiliary placeholder for general loss
        self.z_ph = tf.concat([self.model.z_rec_loss, self.model.z_lin_loss], axis=1)

        # passing variational posteriori to this class
        self.vp_dict = self.model.vp_dict

        # transfer the session to this model
        self.sess = self.model.sess

        # transfer graph to this model
        self.graph = self.model.graph

        # verbose
        self.verbose = edward_configDict['verbose']

        # mode
        self.mode = edward_configDict['mode']

        # mode on whether or not using LRAN
        self.LRAN_mode = LRAN_mode

        # directory
        self.dir = self.model.dir

        # mc sample
        self.mc_sample = mc_sample

        # initialization mode
        self.init_mode = edward_configDict['init']
        self.MAP_dir = edward_configDict['MAP_dir']
        self.init_std_softplus = edward_configDict['init_std_softplus']

        # optimizer
        self.optimizer = 'adam'

    def set_up_K_from_vpdict(self):
        """get K variational posteriori distribution from :attr:`vp_dict` """

        if self.model.model_params['nsd'] == 1:
            # anti-symmetric and diagonal parameterization
            self.K = {'K_K_X': self.vp_dict['K_K_X'],
                      'K_SD': self.vp_dict['K_SD']}
        else:
            # use full matrix of K
            self.K = {'K': self.vp_dict['K']}

    def set_VI(self):
        """set up variational inference with likelihood, priors, so we can infer posterior later"""

        with self.graph.as_default():
            with self.sess.as_default():

                if self.mode == 'MAPSimple':

                    # MAPSimple: setup likelihood with fixed noise

                    output_size = self.model.model_params['structureEncoder'][0]
                    koopman_size = self.model.model_params['structureEncoder'][-1]

                    if self.LRAN_mode == None:

                        print('set up VI... with differential form')
                        total_likelihood_size = output_size + koopman_size

                    else:

                        print('set up VI... with recurrent form')
                        total_likelihood_size = output_size*self.model.T +  koopman_size*(self.model.T-1)

                    # build up a likelihood for general type of loss
                    # with fixed noise variance as 0.001

                    Z_noise_std = 1e-3
                    print('current mode = ', self.mode, ' noise std = ', Z_noise_std)

                    self.z = MultivariateNormalDiag(
                        loc=tf.concat([self.residual_vector_rec_loss,
                                       self.residual_vector_lin_loss], axis=1),
                        scale_diag=tf.constant(Z_noise_std, dtype=TF_FLOAT)*tf.ones(total_likelihood_size,
                                                                              dtype=TF_FLOAT))

                    # link prior with PointMass VP
                    self.inference_dict = {}

                    # 1. link prior_dict <--> vp_dict
                    for key in self.model.prior_dict.keys():
                        print('prior-vp pair linking... ', key)
                        self.inference_dict[self.model.prior_dict[key]] = self.vp_dict[key]

                    # save printing to file
                    with open(self.dir + "/cfg_advi_posteriori.txt", "w") as f:
                        print >> f, ''
                        print >> f, 'Total number of pairs of VI: ', len(self.inference_dict)
                        print >> f, 'VI has been set: with variational posteriori as:'
                        print >> f, ''
                        for vp in self.vp_dict.keys():
                            print >> f, '--'
                            print >> f, 'name = ', vp
                            print >> f, 'shape = ', self.vp_dict[vp].shape
                            print >> f, 'posterior = ', self.vp_dict[vp]

                    # setup MAP inference
                    self.inference = ed.MAP(self.inference_dict, data={self.z: self.z_ph})

                elif self.mode == 'ADVInoARD':

                    # ADVInoARD: setup likelihood with fixed likelihood noise

                    ADVInoARD_noise_scale = tf.constant(1e-3, dtype=TF_FLOAT)

                    output_size = self.model.model_params['structureEncoder'][0]
                    koopman_size = self.model.model_params['structureEncoder'][-1]

                    if self.LRAN_mode == None:

                        print('set up VI... with differential form')
                        total_likelihood_size = output_size + koopman_size

                    else:

                        print('set up VI... with recurrent form')
                        total_likelihood_size = output_size*self.model.T +  koopman_size*(self.model.T-1)

                    # build up a likelihood for general type of loss
                    # with fixed noise variance
                    # small fixed noise forces to fit the data well.

                    error_z_loc = tf.concat([self.residual_vector_rec_loss,
                                             self.residual_vector_lin_loss],
                                            axis=1, name='error_z_loc')

                    # likelihoood is simply a 0.001^2 variance...

                    error_z_scale = ADVInoARD_noise_scale * tf.ones(total_likelihood_size, dtype=TF_FLOAT)

                    print('current mode = ', self.mode, ' noise std = ', ADVInoARD_noise_scale)
                    self.z = MultivariateNormalDiag(loc=error_z_loc, scale_diag=error_z_scale)

                    # 5. construct the inference
                    # -- link prior: prior_dict, hpp_dict <--> vp_dict

                    self.inference_dict = {}

                    # -- 1. link prior_dict <--> vp_dict

                    for key in self.model.prior_dict.keys():
                        print('prior-vp pair linking... ', key)
                        self.inference_dict[self.model.prior_dict[key]] = self.vp_dict[key]

                    # save printing to file
                    with open(self.dir + "/cfg_advi_posteriori.txt", "w") as f:
                        print >> f, ''
                        print >> f, 'Total number of pairs of VI: ', len(self.inference_dict)
                        print >> f, 'VI has been set: with variational posteriori as:'
                        print >> f, ''
                        for vp in self.vp_dict.keys():
                            print >> f, '--'
                            print >> f, 'name = ', vp
                            print >> f, 'shape = ', self.vp_dict[vp].shape
                            print >> f, 'posterior = ', self.vp_dict[vp]

                    # 6. set up the reparameterization part
                    # -- set the new data as auxilary data, we will regard that as our likelihood.

                    self.inference = ed.ReparameterizationKLqp(self.inference_dict, data={self.z: self.z_ph})

                elif self.mode == 'ADVIARD':

                    # 1. set up prior for noise variance in the likelihood of
                    # -- reconstruction
                    # -- linear dynamics

                    if self.LRAN_mode == None:

                        print('set up VI... with differential form')
                        output_shape = self.model.model_params['structureEncoder'][0]
                        koopman_shape = self.model.model_params['structureEncoder'][-1]

                    else:

                        print('set up VI... with recurrent form')
                        output_shape = self.model.model_params['structureEncoder'][0]*self.model.T
                        koopman_shape = self.model.model_params['structureEncoder'][-1]*(self.model.T-1)

                    # setup noise prior for rec and linear error as `HalfCauchy`
                    noise_rec = self.model._create_half_cauchy_prior([self.model.ALPHA_HPP] * tf.ones(output_shape, dtype=TF_FLOAT),
                                                                      [self.model.BETA_HPP] * tf.ones(output_shape, dtype=TF_FLOAT))

                    noise_linear_koopman = self.model._create_half_cauchy_prior(
                                                        [self.model.ALPHA_HPP] * tf.ones(koopman_shape, dtype=TF_FLOAT),
                                                        [self.model.BETA_HPP] * tf.ones(koopman_shape, dtype=TF_FLOAT))

                    # 2. construct the final likelihood
                    # -- residual of reconstruction  ~ N(0, noise_rec)
                    # -- residual of linear dynamics ~ N(0, noise_linear_koopman)
                    error_z_loc = tf.concat([self.residual_vector_rec_loss, self.residual_vector_lin_loss],
                                            axis=1, name='error_z_loc')
                    error_z_scale = tf.concat([tf.sqrt(noise_rec), tf.sqrt(noise_linear_koopman)],
                                              axis=0, name='error_z_scale')

                    print('error_z_loc = ', error_z_loc)
                    print('error_z_scale = ', error_z_scale)

                    self.z = MultivariateNormalDiag(loc=error_z_loc, scale_diag=error_z_scale)


                    # 3. (ALREADY SETUP) variational posteriori in self.vp_dict
                    # -- q w
                    # -- q b
                    # -- q K
                    # -- q scale w
                    # -- q scale b
                    # -- q scale K

                    # 4. variational posteriori in likelihood variance:
                    # note that it is applying softplus scale already! we use log normal
                    # also note that it is a positive support distribution, since it use exp to map to R+

                    # -- qnoise_rec
                    self.vp_dict['scale_noise_rec'] = ed.models.TransformedDistribution(
                        distribution=ed.models.NormalWithSoftplusScale(
                            loc=tf.get_variable('scale_noise_rec/loc', output_shape,
                                                initializer=tf.constant_initializer(-10., dtype=TF_FLOAT),
                                                dtype=TF_FLOAT),
                            scale=tf.get_variable('scale_noise_rec/scale', output_shape,
                                                  initializer=tf.constant_initializer(-10, dtype=TF_FLOAT),
                                                  dtype=TF_FLOAT)),
                        bijector=bijectors.Exp(),
                        name='q_noise_rec')

                    # -- qnoise_linear_koopman
                    self.vp_dict['scale_noise_lin'] = ed.models.TransformedDistribution(
                        distribution=ed.models.NormalWithSoftplusScale(
                            loc=tf.get_variable('scale_noise_lin/loc', koopman_shape,
                                                initializer=tf.constant_initializer(-10., dtype=TF_FLOAT),
                                                dtype=TF_FLOAT),
                            scale=tf.get_variable('scale_noise_lin/scale', koopman_shape,
                                                  initializer=tf.constant_initializer(-10, dtype=TF_FLOAT),
                                                  dtype=TF_FLOAT)),
                        bijector=bijectors.Exp(),
                        name='q_noise_linear_koopman')

                    # 5. construct the inference
                    # -- link prior: prior_dict, hpp_dict <--> vp_dict
                    self.inference_dict = {}
                    # -- 1. link prior_dict <--> vp_dict
                    for key in self.model.prior_dict.keys():
                        print('prior-vp pair linking... ', key)
                        self.inference_dict[self.model.prior_dict[key]] = self.vp_dict[key]

                    # -- 2. link hpp_dict <--> vp_dict
                    for key in self.model.hpp_dict.keys():
                        if 'layer' not in key:
                            print('hyp-prior-vp pair linking... ', key)
                            self.inference_dict[self.model.hpp_dict[key]] = self.vp_dict['scale_' + key]
                    # -- 3. link the noise term with the corresponding VP
                    self.inference_dict[noise_rec] = self.vp_dict['scale_noise_rec']
                    self.inference_dict[noise_linear_koopman] = self.vp_dict['scale_noise_lin']

                    # save printing to file
                    with open(self.dir + "/cfg_advi.txt", "w") as f:
                        print >> f, ''
                        print >> f, 'Total number of pairs of VI: ',len(self.inference_dict)
                        print >> f, 'VI has been set: with variational posteriori as:'
                        print >> f, ''
                        for vp in self.vp_dict.keys():
                            print >> f, '--'
                            print >> f, 'prior = ', vp
                            print >> f, 'shape = ',self.vp_dict[vp].shape
                            print >> f, 'posterior = ', self.vp_dict[vp]

                    # 6. set up the reparameterization part
                    # -- set the new data as auxilary data, we will regard that as our likelihood.
                    self.inference = ed.ReparameterizationKLqp(self.inference_dict, data={self.z: self.z_ph})

                # 7. set up a tensor to evaluate the residual from VP
                self.z_post = ed.copy(self.z, self.inference_dict)
                self.residual_vector_rec_loss_post = ed.copy(self.residual_vector_rec_loss, self.inference_dict)
                self.residual_vector_lin_loss_post = ed.copy(self.residual_vector_lin_loss, self.inference_dict)

    def choose_optimizer(self, optimizer_name):
        """choose the optimizer given optimizer names

        Note:

            1. `adam`
            2. `adadelta`
            3. `adagrad`
            4. `momentum`
            5. `ftrl`
            6. `rmsprop`

        Args:
            optimizer_name (:obj:`str`): the string for optimizer to choose.

        Returns:

            :obj:`class` : optimizer.

        """
        with self.graph.as_default():
            with self.sess.as_default():

                # 'adadelta':
                # 'adagrad':
                # 'momentum':
                # 'adam':
                # 'ftrl':
                # 'rmsprop':

                if optimizer_name == 'adam':

                    # optimizer = L4.L4Adam(fraction=0.2)

                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.model.model_params['learningRate'],
                        epsilon=self.model.model_params['decay'])

                elif optimizer_name == 'adadelta':

                    optimizer = tf.train.AdadeltaOptimizer(
                        learning_rate=self.model.model_params['learningRate'])

                elif optimizer_name == 'adagrad':

                    optimizer = tf.train.AdagradOptimizer(
                        learning_rate=self.model.model_params['learningRate'])

                elif optimizer_name == 'momentum':

                    optimizer = tf.train.MomentumOptimizer(
                        learning_rate=self.model.model_params['learningRate'],
                        momentum=0.9,
                        use_nesterov=False)

                elif optimizer_name == 'ftrl':

                    optimizer = tf.train.FtrlOptimizer(
                        learning_rate=self.model.model_params['learningRate'])

                elif optimizer_name == 'rmsprop':

                    optimizer = tf.train.RMSPropOptimizer(
                        learning_rate=self.model.model_params['learningRate'])

                else:

                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.model.model_params['learningRate'],
                        epsilon=self.model.model_params['decay'])

        return optimizer


    def train(self):
        """training for the Bayesisan DL Koopman model

        Returns:
            :obj:`list` : total MSE list.

        """

        # we set up the K variatioal posteriori from the :attr:`vp_dict`.

        self.set_up_K_from_vpdict()

        optimizer = self.choose_optimizer(self.optimizer)

        with self.graph.as_default():

            with self.sess.as_default():

                if self.LRAN_mode == None:

                    # differential form Bayesian KOopman

                    # using minibatch training for the KLqp
                    self.trainer = edwardMBTrainer(mode=self.mode,
                                                   inference=self.inference,
                                                   n_epoch=self.model.model_params['numberEpoch'],
                                                   n_samples=self.mc_sample,
                                                   batch_size=self.model.model_params['miniBatch'],
                                                   optimizer=optimizer,
                                                   x=self.model.X,
                                                   y=self.model.Xdot,
                                                   z=self.z,  # likelihood RV
                                                   z_ph=self.z_ph,  # placeholder to let data flow in from true
                                                   z_post=self.z_post,  # placeholder to compute MSE
                                                   logstr=self.dir + '/log',
                                                   sess=self.sess,
                                                   K=self.K,
                                                   init_mode=self.init_mode,
                                                   init_std_softplus=self.init_std_softplus,
                                                   MAP_dir=self.MAP_dir,
                                                   residual_lin_loss_post=self.residual_vector_lin_loss_post,
                                                   residual_rec_loss_post=self.residual_vector_rec_loss_post,
                                                   nsd_mode=self.model.model_params['nsd'])

                    self.model.initilize()

                    # using validation data as testing data
                    self.trainer.set_data(x_train=self.model.Xtrain, x_test=self.model.Xvalid,
                                          y_train=self.model.XdotTrain, y_test=self.model.XdotValid)

                else:

                    # recurrent form Bayesian Koopman

                    # using minibatch training for the KLqp
                    self.trainer = edwardMBTrainerLRAN(mode=self.mode,
                                                       inference=self.inference,
                                                       n_epoch=self.model.model_params['numberEpoch'],
                                                       n_samples=self.mc_sample,
                                                       batch_size=self.model.model_params['miniBatch'],
                                                       optimizer=optimizer,
                                                       x=self.model.X,
                                                       future_X_list=self.model.future_X_list,
                                                       z=self.z,  # likelihood RV
                                                       z_ph=self.z_ph,  # placeholder to let data flow in from true
                                                       z_post=self.z_post,  # placeholder to compute MSE
                                                       logstr=self.dir + '/log''log',
                                                       sess=self.sess,
                                                       K=self.K,
                                                       init_mode=self.init_mode,
                                                       init_std_softplus=self.init_std_softplus,
                                                       MAP_dir=self.MAP_dir,
                                                       residual_lin_loss_post=self.residual_vector_lin_loss_post,
                                                       residual_rec_loss_post=self.residual_vector_rec_loss_post,
                                                       nsd_mode=self.model.model_params['nsd'])

                    self.model.initilize()

                    # using validation data as testing data
                    self.trainer.set_data(x_train=self.model.Xtrain, x_future_train=self.model.XfutureTrain,
                                          x_valid=self.model.Xvalid, x_future_valid=self.model.XfutureValid)

                # start to training in a minibatch fashion
                total_MSE_list = self.trainer.minibatch_train()

                # print 'total_MSE_list =', total_MSE_list

        return total_MSE_list

    def update_graph(self):
        self.graph = self.trainer.graph
        self.model.graph = self.graph

        self.model.sess = self.trainer.sess
        self.sess = self.trainer.sess

    def save_model(self):
        """save model calling the ``save_model`` function in ``ClassDLKoopmanCont`` or
        ``classDLKoopmanLRAN`` """

        with self.graph.as_default():
            with self.sess.as_default():
                # saving mode using DL Koopman
                self.model.save_model()













class edwardMBTrainer(object):
    """Class of calling ``edward`` to do ADVI for ``differential form``.

    Args:

        mode (:obj:`str`): Selecting the the type of MFVI framework to use.

        inference (:obj:`class`): ``edward.Inference`` class instance. It is the main object for variational inference.

        n_epoch (:obj:`int`): number of epoch.

        n_samples (:obj:`int`): number of total sampling training data.

        batch_size (:obj:`int`): batch size for mini-batch training.

        optimizer (:obj:`class`): an optimizer for training. ``tf.train.Adam`` can also be used.

        x (:obj:`tf.Tensor`): placeholder for feeding the input data, i.e., :math:`x`.

        y (:obj:`tf.Tensor`): placeholder for feeding the input data, i.e., :math:`\dot{x}`.

        z (:obj:`tf.Tensor`): likelihood tensor just to construct the graph.

        z_ph (:obj:`tf.Tensor`): placeholder for feeding the likelihood, i.e., zeros.

        z_post (:obj:`tf.Tensor`): after copying the posteriori to the true parameters, this is the output placeholder
            that enables one to evaluate the output of the z.

        logstr (:obj:`str`): the path to write logs from Edward. Not used anymore.

        sess(:obj:`class`): A context manager using this session as the default session.

        K (:obj:`tf.Tensor`): Feed the tensor ``K`` matrix.

        init_mode (:obj:`str`): The mode we choose to initialize the process of AVDI.

        init_std_softplus (:obj:`float`): the initial standard deviation we used in softplus, the more negative,
            the smaller it is.

        MAP_dir (:obj:`str`): path to the pre-trained model when :attr:`init_mode` = ``nonBayes``.

        residual_lin_loss_post (:obj:`tf.Tensor`): linear dynamics residuals from model construction in the
            class ``ClassDLKoopmanCont`` defined at :ref:`ClassDLKoopmanCont`.

        residual_rec_loss_post (:obj:`tf.Tensor`): reconstruction dynamics residuals from model construction in the
            class ``ClassDLKoopmanCont`` defined at :ref:`ClassDLKoopmanCont`.

    Attributes:
        sess (:obj:`class`): A context manager using this session as the default session.

        graph (:obj:`class`): A `Graph` instance supports an arbitrary number of “collections” that are identified by name.

        mode (:obj:`str`): Selecting the the type of MFVI framework to use.

            Note:
                * ADVInoARD
                * ADVIARD
                * MAPSimple

        inference (:obj:`class`): ``edward.Inference`` class instance. It is the main object for variational inference.

        n_epoch (:obj:`int`): number of epoch.

        n_samples (:obj:`int`): number of total sampling training data.

        batch_size (:obj:`int`): batch size for mini-batch training.

        optimizer (:obj:`class`): an optimizer for training. ``tf.train.Adam`` can also be used.

        logstr (:obj:`str`): the path to write logs from Edward. Not used anymore.

        x (:obj:`tf.Tensor`): placeholder for feeding the input data, i.e., :math:`x`.

        y (:obj:`tf.Tensor`): placeholder for feeding the input data, i.e., :math:`\dot{x}`.

        z (:obj:`tf.Tensor`): likelihood tensor just to construct the graph.

        z_ph (:obj:`tf.Tensor`): placeholder for feeding the likelihood, i.e., zeros.

        z_post (:obj:`tf.Tensor`): after copying the posteriori to the true parameters, this is the output placeholder
            that enables one to evaluate the output of the z.

        K (:obj:`dict`): A collection with keys as``K_K_X`` and ``K_SD`` to map to the probability distribution of those two components. Here we only assume the stabilization form.

        init_mode (:obj:`str`): The mode we choose to initialize the process of AVDI.

            Note:

                * ``Standard``
                    simply initialize the weights just variance-scaling method, with scale initialized very small.
                * ``nonBayes``
                    simply initialize with a fined-tuned deterministic result. This is helpful for training the case of the Duffing oscillator.

        init_std_softplus (:obj:`float`): the initial standard deviation we used in softplus, the more negative ,
            the smaller it is.

        MAP_dir (:obj:`str`): path to the pre-trained model when :attr:`init_mode` = ``nonBayes``.

        residual_lin_loss_post (:obj:`tf.Tensor`): linear dynamics residuals from model construction in the
           class ``ClassDLKoopmanCont`` defined at :ref:`ClassDLKoopmanCont`.

        residual_rec_loss_post (:obj:`tf.Tensor`): reconstruction dynamics residuals from model construction in the
           class ``ClassDLKoopmanCont`` defined at :ref:`ClassDLKoopmanCont`.

        n_data (:obj:`int`): number of training data.

        x_train (:obj:`numpy.ndarray`): training states.

        y_train (:obj:`numpy.ndarray`): training time derivative.

        x_test (:obj:`numpy.ndarray`): testing states.

        y_test (:obj:`numpy.ndarray`): testing time derivative.

        z_train (:obj:`numpy.ndarray`): feeding training likelihood data, i.e., zeros.

        z_test (:obj:`numpy.ndarray`): feediing testing likelihood data, i.e., zeros.

        data (:obj:`generator`): a generator for getting mini-batch training data.

        n_batch (:obj:`int`): number of batches in the total training data.




    """

    def __init__(self, mode, inference, n_epoch, n_samples, batch_size, optimizer,
                 x, y, z, z_ph, z_post, logstr, sess, K,
                 init_mode, init_std_softplus, MAP_dir,
                 residual_lin_loss_post, residual_rec_loss_post,
                 nsd_mode):

        self.sess = sess
        self.graph = self.sess.graph

        # with self.graph.as_default():
        self.mode = mode
        self.inference = inference
        self.n_epoch = n_epoch
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.logstr = logstr
        self.nsd_mode = nsd_mode
        self.mc_samples = 500 # for mean evaluation

        # input placeholder --> feed training feature in

        self.x = x
        self.y = y

        # likelihood tensor --> to construct the graph

        self.z = z

        # output placeholder --> feed training target in

        self.z_ph = z_ph

        # tensor for evaluating the error of residual

        self.z_post = z_post

        self.K = K
        self.init_mode = init_mode
        self.init_std_softplus = init_std_softplus
        self.MAP_dir = MAP_dir

        self.residual_lin_loss_post = residual_lin_loss_post
        self.residual_rec_loss_post = residual_rec_loss_post

        # create log file
        self.loss_log_file = open(self.logstr + '.txt-bayesian', 'w+')

    def generator(self, arrays, batch_size):
        """Generator for mini-batches

        The batches are generated with respect to each array's first axis.

        Examples:

            ::

                self.data = self.generator([self.x_train, self.y_train], self.batch_size)


        Args:
            arrays (:obj:`numpy.ndarray`): training data with first axis as number of samples.

            batch_size (:obj:`int`): batch size in order to divide data to get mini-batches generator.

        Yield:
            :obj:`numpy.ndarray` : next batch size data.

        """

        starts = [0] * len(arrays)  # pointers to where we are in iteration
        while True:
            batches = []
            for i, array in enumerate(arrays):
                start = starts[i]
                stop = start + batch_size
                diff = stop - array.shape[0]
                if diff <= 0:
                    batch = array[start:stop]
                    starts[i] += batch_size
                else:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                batches.append(batch)
            yield batches

    def set_data(self, x_train, y_train, x_test, y_test):
        """Set up data for variational inference process.

        Args:

            x_train (:obj:`numpy.ndarray`): training states, i.e., :math:`x`.

            y_train (:obj:`numpy.ndarray`): training time derivative, i.e., :math:`\dot{x}`

            x_test (:obj:`numpy.ndarray`): testing states

            y_test (:obj:`numpy.ndarray`): testing time derivative

        """

        self.n_data = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # adding z_test auxially variables.
        # Note:
        #   they are zeros

        self.z_train = np.zeros((y_train.shape[0], self.z_ph.get_shape()[1]))
        self.z_test  = np.zeros((y_test.shape[0],  self.z_ph.get_shape()[1]))

        # minibatch generator from training data pairs.

        self.data = self.generator([self.x_train, self.y_train], self.batch_size)

        # number of batches in the training data

        self.n_batch = int(self.n_data / self.batch_size)

    def sample_K_and_return_mean(self):
        """MC sampling the variational posteriori for Koopman operator ``K`` then return the mean.

        We just do a Monte carlo sampling with number as :attr:`mc_samples`. Default value as 100.
        Note that we simply assumes we using the stabilized form.

        Attributes:
            mc_samples (:obj:`int`): number of Monte carlo samples for getting ``K``.

        Returns:
            :obj:`numpy.ndarray` : mean Koopman operator matrix.

        """

        # getting the probability distribution from :attr:`K`.

        if self.nsd_mode == 1:

            K_K_X = self.K['K_K_X']
            K_SD = self.K['K_SD']

            with self.sess.as_default():
                samples_X_upper = tf.matrix_band_part(tf.reduce_mean(K_K_X.sample(self.mc_samples),axis=0), 0, 1)
                samples_K_XX = samples_X_upper - tf.transpose(samples_X_upper)
                samples_K_SD = tf.diag(tf.reduce_mean(K_SD.sample(self.mc_samples),axis=0))
                samples_K = self.sess.run(samples_K_XX - samples_K_SD)

        else:

            K = self.K['K']

            with self.sess.as_default():
                samples_K = self.sess.run(tf.reduce_mean(K.sample(self.mc_samples), axis=0))

        return samples_K

    def minibatch_train(self):
        """mini-batch training function

        Returns:
            :obj:`list` : total MSE list
        """

        with self.graph.as_default():
            with self.sess.as_default():

                # the scaling is to cheat on the optimizer to make it feels like seeing a whole set of data
                if self.mode == 'MAPSimple':

                    self.inference.initialize(optimizer=self.optimizer,
                                              n_iter=self.n_batch * self.n_epoch,
                                              scale={self.z: tf.constant(self.n_data *1.0 / self.batch_size, TF_FLOAT)},
                                              logdir=None,
                                              auto_transform=True)

                else:

                    self.inference.initialize(optimizer=self.optimizer,
                                              n_iter=self.n_batch * self.n_epoch,
                                              n_samples=self.n_samples,
                                              scale={self.z: tf.constant(self.n_data *1.0 / self.batch_size, TF_FLOAT)},
                                              logdir=None,
                                              auto_transform=True)

                ## initialize the encoder, decoder, K as well

                self.sess.run(tf.global_variables_initializer())

                if self.init_mode == 'Standard':

                    print 'ADVI initialized in the classical way (no pre-trained parameters)'

                else:

                    print 'ADVI initialized in nonBayes: for K, encoder, decoder'

                    # still initlize other things

                    para_dict = np.load(self.MAP_dir + '/saved_para.npy')
                    para_dict = para_dict.item()

                    # use tf.assign for the loc in K, decoder, encoder
                    for key in para_dict.keys():

                        print 'key = ', key, ' is initialized manually! '

                        if key == 'K':

                            if self.nsd_mode == 1:

                                # first consider the K_X part
                                tv = self.graph.get_tensor_by_name('qK_K_X/loc:0')

                                print type(tv), type(para_dict[key])

                                # get the full K from the nonBayes result
                                K = para_dict[key]

                                print('validating the pretrained K...')
                                print('eig of K = ', np.linalg.eig(K)[0])

                                # get K's upper diagonal
                                K_upper = np.diagonal(K, offset=1)
                                K_K_X = np.diag(K_upper, k=1)

                                print('K = ', K)
                                print('K_K_X should be only the offset of K', K_K_X)

                                # get K's diagonal, but it is a minius one...
                                K_SD = - np.diagonal(K, offset=0)

                                # assign the nonBayes K_K_X to the Bayes
                                self.sess.run(tf.assign(tv, K_K_X))

                                # assign the nonBayes positive square diagonal part to SD part
                                # Note that qK_SD/loc:0 is inside a lognormal, so one need  to take a log
                                tv = self.graph.get_tensor_by_name('qK_SD/loc:0')

                                # prevent nan
                                K_SD += 1e-6

                                # Becase K_SD is vari-posterioed with a log normal
                                self.sess.run(tf.assign(tv, tf.log(K_SD)))

                            else:

                                # if full matrix K is used

                                tv = self.graph.get_tensor_by_name('qK/loc:0')

                                K = para_dict[key]

                                self.sess.run(tf.assign(tv, K))

                            # Now consider the assigning ``scale`` part --> make them small
                            if self.mode == 'MAPSimple':

                                pass

                            elif self.mode == 'ADVInoARD':

                                # Here, we force initial scale to be quite negative, note that
                                # all the scale parameter is transformed by a softmax to make it positive, so we just assign
                                # very negative to make the variance of posteriori extremely small.

                                if self.nsd_mode == 1:

                                    tv = self.graph.get_tensor_by_name('qK_K_X/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qK_SD/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                else:

                                    tv = self.graph.get_tensor_by_name('qK/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                            elif self.mode == 'ADVIARD':

                                if self.nsd_mode == 1:

                                    tv = self.graph.get_tensor_by_name('qK_K_X/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qK_SD/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_K_X/loc:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_K_X/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_SD/loc:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_SD/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                else:

                                    tv = self.graph.get_tensor_by_name('qK/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                        else:

                            ## now let's put the rest: encoder and decoder weights

                            if self.mode == 'MAPSimple':

                                tv = self.graph.get_tensor_by_name(key + '/loc:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))


                            elif self.mode == 'ADVInoARD':

                                print('use pretrained weight and biases...', key)

                                tv = self.graph.get_tensor_by_name(key + '/loc:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))

                                # debug: reduce stddev initial to zero

                                tv = self.graph.get_tensor_by_name(key + '/scale:0')
                                self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))


                            elif self.mode == 'ADVIARD':

                                tv = self.graph.get_tensor_by_name(key + '/loc_1:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))

                                tv = self.graph.get_tensor_by_name(key + '/scale_1:0')
                                self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                # make sure the scale in prior is also near zero, which means prior is like a point mass
                                tv = self.graph.get_tensor_by_name('qscale_' + key + '/loc:0')
                                self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                tv = self.graph.get_tensor_by_name('qscale_' + key + '/scale:0')
                                self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                # tv = self.graph.get_tensor_by_name(key + '/loc_1:0')
                                #
                                # print type(tv), type(para_dict[key])
                                #
                                # self.sess.run(tf.assign(tv, para_dict[key]))
                                #
                                # # debug: reduce stddev initial to zero
                                # tv = self.graph.get_tensor_by_name(key + '/scale_1:0')
                                #
                                # self.sess.run(tf.assign(tv, tf.ones(tv.get_shape()) * self.init_std_softplus))

                            else:

                                raise NotImplementedError

                ## before training, check initial error
                res_train = self.sess.run(self.z_post.mean(), feed_dict={self.x: self.x_train,
                                                                  self.y: self.y_train,
                                                                  self.z_ph: self.z_train})
                train_MSE = np.square(res_train).mean()

                res_test = self.sess.run(self.z_post.mean(), feed_dict={self.x: self.x_test,
                                                                 self.y: self.y_test,
                                                                 self.z_ph: self.z_test})
                valid_MSE = np.square(res_test).mean()

                print self.sess.run(self.residual_lin_loss_post, feed_dict={self.x: self.x_train,
                                                                            self.y: self.y_train,
                                                                            self.z_ph: self.z_train})

                print self.sess.run(self.residual_rec_loss_post, feed_dict={self.x: self.x_train,
                                                                            self.y: self.y_train,
                                                                            self.z_ph: self.z_train})

                print 'MAP train_MSE = ', train_MSE
                print 'MAP valid_MSE = ', valid_MSE


                ## batch training begin
                tot_batch_loss = 0

                self.total_MSE_list = {}
                train_MSE_list = []
                valid_MSE_list = []

                for _ in xrange(self.inference.n_iter):

                    ## each iteration is processing one batch

                    # generates the batch!!!
                    X_batch, y_batch = next(self.data)

                    ## update with the data!
                    ## it is feeding both X and y
                    # info_dict = self.inference.update({x: X_batch, y_ph: y_batch})

                    ## change it with the generalized loss
                    z_batch = np.zeros(( y_batch.shape[0], self.z_ph.get_shape()[1] ))
                    info_dict = self.inference.update({self.x: X_batch, self.y: y_batch, self.z_ph: z_batch})

                    # inference.print_progress(info_dict)

                    # summing loss
                    tot_batch_loss += info_dict['loss']

                    # then we finished with epoch
                    if _ % (self.n_batch*50) == 0:

                        res_train_mean = self.sess.run(self.z_post.mean(), feed_dict={self.x: self.x_train, self.y: self.y_train,
                                                                                self.z_ph: self.z_train})
                        train_MSE_mean = np.square(res_train_mean).mean()

                        res_test_mean = self.sess.run(self.z_post.mean(), feed_dict={self.x: self.x_test, self.y: self.y_test,
                                                                                self.z_ph: self.z_test})
                        valid_MSE_mean = np.square(res_test_mean).mean()

                        res_train = self.sess.run(self.z_post,
                                                  feed_dict={self.x: self.x_train, self.y: self.y_train,
                                                             self.z_ph: self.z_train})
                        train_MSE = np.square(res_train).mean()

                        res_test = self.sess.run(self.z_post,
                                                 feed_dict={self.x: self.x_test, self.y: self.y_test,
                                                            self.z_ph: self.z_test})
                        valid_MSE = np.square(res_test).mean()

                        # train_MSE = ed.evaluate('mean_squared_error',
                        #                                   data={self.x: self.x_train,
                        #                                         self.y: self.y_train,
                        #                                         self.z_post: self.z_train})
                        #
                        # valid_MSE = ed.evaluate('mean_squared_error',
                        #                                  data={self.x: self.x_test,
                        #                                        self.y: self.y_test,
                        #                                        self.z_post: self.z_test})

                        print ''
                        print 'current epoch = ', int(_ / self.n_batch), ' total epoch = ', self.n_epoch
                        print 'mean of error residual = ', train_MSE_mean, valid_MSE_mean
                        print 'train MSE = ', train_MSE
                        print 'validation MSE = ', valid_MSE

                        ## evaluate the Koopman eigenvalues for the mean of the Koopman operator

                        # sample K to get mean value
                        K_mean = self.sample_K_and_return_mean()
                        evalue = np.linalg.eig(K_mean)[0]
                        print 'eigenvalues = ', evalue

                        ## write all the stuff into log so I can monitor them on cluster
                        self.loss_log_file.write('' + '\n')
                        self.loss_log_file.write('=============================================')
                        self.loss_log_file.write('current epoch = ' + str(int(_ / self.n_batch)) + ' total epoch = ' + str(self.n_epoch) + '\n')
                        self.loss_log_file.write('mean of error residual = ' + str(train_MSE_mean) + str(valid_MSE_mean) + '\n')
                        self.loss_log_file.write('train MSE = ' + str(train_MSE) + '\n')
                        self.loss_log_file.write('validation MSE = ' + str( valid_MSE) + '\n')
                        self.loss_log_file.write('eigenvalues = ' + str(evalue) + '\n')

                        train_MSE_list.append(train_MSE)
                        valid_MSE_list.append(valid_MSE)

                self.total_MSE_list['train'] = train_MSE_list
                self.total_MSE_list['valid'] = valid_MSE_list

                # end inference.
                self.inference.finalize()

                # close loss log file
                self.loss_log_file.close()

                # print 'total_mse =', self.total_MSE_list

        return self.total_MSE_list

    def get_graph(self):
        return self.graph


class edwardMBTrainerLRAN(object):
    """Class of calling ``edward`` to do ADVI for ``recurrent form``

    The only difference with :class:`edwardMBTrainer` is that :math:`\dot{x}` is not used and we use ``T`` as the time delay steps.

    """

    def __init__(self, mode, inference, n_epoch, n_samples, batch_size, optimizer,
                 x, future_X_list, z, z_ph, z_post, logstr, sess, K,
                 init_mode, init_std_softplus, MAP_dir,
                 residual_lin_loss_post, residual_rec_loss_post,
                 nsd_mode):

        self.sess = sess
        self.graph = self.sess.graph

        # with self.graph.as_default():
        self.mode = mode
        self.inference = inference
        self.n_epoch = n_epoch
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.logstr = logstr
        self.nsd_mode = nsd_mode
        self.mc_samples = 100 # for mean evaluation

        # input placeholder --> feed training feature in
        self.x = x
        self.future_X_list = future_X_list

        # likelihood tensor --> to construct the graph
        self.z = z

        # output placeholder --> feed training target in
        self.z_ph = z_ph

        # tensor for evaluating the error of residual
        self.z_post = z_post

        self.K = K
        self.init_mode = init_mode
        self.init_std_softplus = init_std_softplus
        self.MAP_dir = MAP_dir
        self.residual_lin_loss_post = residual_lin_loss_post
        self.residual_rec_loss_post = residual_rec_loss_post

        # create log file
        self.loss_log_file = open(self.logstr + '.txt-bayesian-lran', 'w')

        return

    def generator(self, arrays, batch_size):
        """Generate batches, one with respect to each array's first axis."""
        starts = [0] * len(arrays)  # pointers to where we are in iteration
        while True:
            batches = []
            for i, array in enumerate(arrays):
                start = starts[i]
                stop = start + batch_size
                diff = stop - array.shape[0]
                if diff <= 0:
                    batch = array[start:stop]
                    starts[i] += batch_size
                else:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                batches.append(batch)
            yield batches

    def set_data(self, x_train, x_future_train,x_valid, x_future_valid):
        self.n_data = x_train.shape[0]
        self.x_train = x_train
        self.x_valid = x_valid
        self.x_future_train = x_future_train
        self.x_future_valid = x_future_valid


        ## adding z_test auxially variables.
        # -- they are zeros
        self.z_train = np.zeros((x_train.shape[0], self.z_ph.get_shape()[1]))
        self.z_test = np.zeros((x_valid.shape[0], self.z_ph.get_shape()[1]))

        # minibatch generator
        self.data = self.generator([self.x_train, self.x_future_train], self.batch_size)

        self.n_batch = int(self.n_data / self.batch_size)

    def sample_K_and_return_mean(self):

        if self.nsd_mode == 1:
            K_K_X = self.K['K_K_X']
            K_SD = self.K['K_SD']

            with self.sess.as_default():
                samples_X_upper = tf.matrix_band_part(tf.reduce_mean(K_K_X.sample(self.mc_samples), axis=0), 0, 1)
                samples_K_XX = samples_X_upper - tf.transpose(samples_X_upper)
                samples_K_SD = tf.diag(tf.reduce_mean(K_SD.sample(self.mc_samples), axis=0))
                samples_K = self.sess.run(samples_K_XX - samples_K_SD)

        else:

            K = self.K['K']
            with self.sess.as_default():
                samples_K = self.sess.run(tf.reduce_mean(K.sample(self.mc_samples),axis=0))

        return samples_K

    def minibatch_train(self):


        with self.graph.as_default():
            with self.sess.as_default():

                # the scaling is to cheat on the optimizer to make it feels like seeing a whole set of data
                if self.mode == 'MAPSimple':

                    self.inference.initialize(optimizer=self.optimizer,
                                              n_iter=self.n_batch * self.n_epoch,
                                              scale={
                                                  self.z: tf.constant(self.n_data * 1.0 / self.batch_size, TF_FLOAT)},
                                              logdir=None,
                                              auto_transform=True)

                else:

                    self.inference.initialize(optimizer=self.optimizer,
                                              n_iter=self.n_batch * self.n_epoch,
                                              n_samples=self.n_samples,
                                              scale={self.z: self.n_data * 1.0 / self.batch_size},
                                              logdir=None,
                                              auto_transform=True)

                ## initialize the encoder, decoder, K as well
                self.sess.run(tf.global_variables_initializer())

                if self.init_mode == 'Standard':

                    print 'ADVI initialized in the standard way'

                else:

                    print 'ADVI initialized in nonBayes: for K, encoder, decoder'

                    # still initlize other things

                    para_dict = np.load(self.MAP_dir + '/saved_para.npy')
                    para_dict = para_dict.item()

                    # use tf.assign for the loc in K, decoder, encoder
                    for key in para_dict.keys():

                        print 'key = ', key, ' is initialized manually! '

                        if key == 'K':

                            if self.nsd_mode == 1:

                                # first consider the K_X part
                                tv = self.graph.get_tensor_by_name('qK_K_X/loc:0')

                                print type(tv), type(para_dict[key])

                                # get the full K from the nonBayes result
                                K = para_dict[key]

                                # get K's upper diagonal
                                K_upper = np.diagonal(K, offset=1)
                                K_K_X = np.diag(K_upper, k=1)

                                # get K's diagonal, but it is a minius one...
                                K_SD = - np.diagonal(K, offset=0)

                                # assign the nonBayes K_K_X to the Bayes
                                self.sess.run(tf.assign(tv, K_K_X))

                                # assign the nonBayes positive square diagonal part to SD part
                                # Note that qK_SD/loc:0 is inside a lognormal, so one need  to take a log
                                tv = self.graph.get_tensor_by_name('qK_SD/loc:0')
                                self.sess.run(tf.assign(tv, tf.log(tf.constant(K_SD, dtype=TF_FLOAT))))

                            else:

                                tv = self.graph.get_tensor_by_name('qK/loc:0')
                                K = para_dict[key]
                                self.sess.run(tf.assign(tv, K))

                            # Now consider the scale part
                            if self.mode == 'MAPSimple':

                                pass

                            elif self.mode == 'ADVInoARD':

                                # Here, we force initial scale to be quite negative, note that
                                # all the scale parameter is transformed by a softmax to make it positive, so we just assign
                                # very negative to make the variance of posteriori extremely small.

                                if self.nsd_mode == 1:

                                    tv = self.graph.get_tensor_by_name('qK_K_X/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qK_SD/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                else:

                                    tv = self.graph.get_tensor_by_name('qK/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                            elif self.mode == 'ADVIARD':

                                if self.nsd_mode == 1:

                                    tv = self.graph.get_tensor_by_name('qK_K_X/scale:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qK_SD/scale:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_K_X/loc:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_K_X/scale:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_SD/loc:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                    tv = self.graph.get_tensor_by_name('qscaleK_SD/scale:0')
                                    self.sess.run(
                                        tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                else:

                                    tv = self.graph.get_tensor_by_name('qK/scale:0')
                                    self.sess.run(tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                        else:

                            ## now let's put the rest: encoder and decoder

                            if self.mode == 'MAPSimple':

                                tv = self.graph.get_tensor_by_name(key + '/loc:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))


                            elif self.mode == 'ADVInoARD':

                                tv = self.graph.get_tensor_by_name(key + '/loc:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))

                                # debug: reduce stddev initial to zero

                                tv = self.graph.get_tensor_by_name(key + '/scale:0')
                                self.sess.run(
                                    tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))


                            elif self.mode == 'ADVIARD':

                                tv = self.graph.get_tensor_by_name(key + '/loc_1:0')
                                # print type(tv), type(para_dict[key])
                                self.sess.run(tf.assign(tv, para_dict[key]))

                                tv = self.graph.get_tensor_by_name(key + '/scale_1:0')
                                self.sess.run(
                                    tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                # make sure the scale in prior is also near zero, which means prior is like a point mass
                                tv = self.graph.get_tensor_by_name('qscale_' + key + '/loc:0')
                                self.sess.run(
                                    tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                tv = self.graph.get_tensor_by_name('qscale_' + key + '/scale:0')
                                self.sess.run(
                                    tf.assign(tv, tf.ones(tv.get_shape(), dtype=TF_FLOAT) * self.init_std_softplus))

                                # tv = self.graph.get_tensor_by_name(key + '/loc_1:0')
                                #
                                # print type(tv), type(para_dict[key])
                                #
                                # self.sess.run(tf.assign(tv, para_dict[key]))
                                #
                                # # debug: reduce stddev initial to zero
                                # tv = self.graph.get_tensor_by_name(key + '/scale_1:0')
                                #
                                # self.sess.run(tf.assign(tv, tf.ones(tv.get_shape()) * self.init_std_softplus))

                            else:

                                raise NotImplementedError

                ## prepare training data as dict for evaluation
                feed_dict_whole_train = {self.x: self.x_train[:, :]}
                for ii in xrange(len(self.future_X_list)):
                    feed_dict_whole_train[self.future_X_list[ii]] = self.x_future_train[:, ii, :]

                ## prepare validation data as dict for evaluation
                feed_dict_whole_valid = {self.x: self.x_valid[:, :]}
                for ii in xrange(len(self.future_X_list)):
                    feed_dict_whole_valid[self.future_X_list[ii]] = self.x_future_valid[:, ii, :]

                feed_dict_whole_train[self.z_ph] = self.z_train
                feed_dict_whole_valid[self.z_ph] = self.z_train

                ## before training, check initial error

                res_train = self.sess.run(self.z_post.mean(), feed_dict=feed_dict_whole_train)
                train_MSE = np.square(res_train).mean()

                res_test = self.sess.run(self.z_post.mean(), feed_dict=feed_dict_whole_valid)
                valid_MSE = np.square(res_test).mean()

                print self.sess.run(self.residual_lin_loss_post, feed_dict=feed_dict_whole_train)

                print self.sess.run(self.residual_rec_loss_post, feed_dict=feed_dict_whole_valid)

                print 'MAP train_MSE = ', train_MSE
                print 'MAP valid_MSE = ', valid_MSE

                ## batch training begin
                tot_batch_loss = 0

                self.total_MSE_list = {}
                train_MSE_list = []
                valid_MSE_list = []

                for _ in xrange(self.inference.n_iter):

                    ## each iteration is processing one batch

                    # generates the batch!!!
                    X_batch, X_future_batch = next(self.data)

                    ## prepare LRAN update dict
                    feed_dict_whole = {self.x: X_batch[:, :]}
                    for ii in xrange(X_future_batch.shape[1]):
                        feed_dict_whole[self.future_X_list[ii]] = X_future_batch[:, ii, :]


                    ## update with the data!
                    ## it is feeding both X and y
                    # info_dict = self.inference.update({x: X_batch, y_ph: y_batch})

                    ## change it with the generalized loss
                    z_batch = np.zeros((X_future_batch.shape[0], self.z_ph.get_shape()[1]))

                    feed_dict_whole[self.z_ph] = z_batch

                    info_dict = self.inference.update(feed_dict_whole)

                    # inference.print_progress(info_dict)

                    # summing loss
                    tot_batch_loss += info_dict['loss']

                    # then we finished with epoch
                    if _ % (self.n_batch * 100) == 0:
                        res_train_mean = self.sess.run(self.z_post.mean(), feed_dict=feed_dict_whole_train)
                        train_MSE_mean = np.square(res_train_mean).mean()

                        res_test_mean = self.sess.run(self.z_post.mean(), feed_dict=feed_dict_whole_valid)
                        valid_MSE_mean = np.square(res_test_mean).mean()

                        res_train = self.sess.run(self.z_post, feed_dict=feed_dict_whole_train)
                        train_MSE = np.square(res_train).mean()

                        res_test = self.sess.run(self.z_post, feed_dict=feed_dict_whole_valid)
                        valid_MSE = np.square(res_test).mean()

                        # train_MSE = ed.evaluate('mean_squared_error',
                        #                                   data={self.x: self.x_train,
                        #                                         self.y: self.y_train,
                        #                                         self.z_post: self.z_train})
                        #
                        # valid_MSE = ed.evaluate('mean_squared_error',
                        #                                  data={self.x: self.x_test,
                        #                                        self.y: self.y_test,
                        #                                        self.z_post: self.z_test})

                        print ''
                        print 'current epoch = ', int(_ / self.n_batch), ' total epoch = ', self.n_epoch
                        print 'mean of error residual = ', train_MSE_mean, valid_MSE_mean
                        print 'train MSE = ', train_MSE
                        print 'validation MSE = ', valid_MSE

                        ## evaluate the Koopman eigenvalues for the mean of the Koopman operator

                        # sample K to get mean value
                        K_mean = self.sample_K_and_return_mean()
                        evalue = np.linalg.eig(K_mean)[0]
                        print 'eigenvalues = ', evalue

                        ## write all the stuff into log so I can monitor them on cluster
                        self.loss_log_file.write('')
                        self.loss_log_file.write('=============================================')
                        self.loss_log_file.write('current epoch = ' + str(int(_ / self.n_batch)) + ' total epoch = ' + str(self.n_epoch))
                        self.loss_log_file.write('mean of error residual = ' + str(train_MSE_mean) + str(valid_MSE_mean))
                        self.loss_log_file.write('train MSE = ' + str(train_MSE))
                        self.loss_log_file.write('validation MSE = ' + str( valid_MSE))
                        self.loss_log_file.write('eigenvalues = ' + str(evalue))

                        train_MSE_list.append(train_MSE)
                        valid_MSE_list.append(valid_MSE)

                self.total_MSE_list['train'] = train_MSE_list
                self.total_MSE_list['valid'] = valid_MSE_list

                # end inference.
                self.inference.finalize()

                # close loss log file
                self.loss_log_file.close()


                # print 'total_mse =', self.total_MSE_list

        return self.total_MSE_list

    def get_graph(self):
        return self.graph



if __name__ == '__main__':
    pass
