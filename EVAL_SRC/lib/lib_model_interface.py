#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of model interface for evaluation module in ``main_apo.py``
"""

import sys
import numpy
sys.dont_write_bytecode = True
import numpy as np
import tensorflow as tf
import os
os.environ['TF_C_API_GRAPH_CONSTRUCTION']='0'

import edward as ed
from edward.models import Normal, MultivariateNormalDiag
from tensorflow.contrib.distributions import bijectors
from MODEL_SRC.lib.model import F_duffing_2d_system, F_simple_2d_system
from MODEL_SRC.lib.utilities import myswish_beta, penalized_tanh
from numpy import linalg as LA
from MODEL_SRC.lib.utilities import TF_FLOAT


class ClassFNN_edward(object):
    """
    Class of Feedforward neural network given topology, structure, activation function, and recurrent or not,
    SVD_DMD or not, to replicate the model we saved.

    Note:

        We did this simply because there is no good save_model functionality in edward.

    Args:
        sess (:obj:`class`): the session we used in tensorflow.

        layer_structure_list (:obj:`list`): the layer structure we used in constructing the Feedforward neural network.

        act_fun_str (:obj:`str`): the name of the activation function to use.

        prefix (:obj:`str`): the prefix used in the weights and biases parameters, either ``dec`` or ``enc``.

        mode (:obj:`str`): the type of ADVI to use, either ``MAPSimple`` or ``ADVInoARD``, ``ADVIARD``.

        LRAN_T (:obj:`int`): the look forward time window length.

        SVD_DMD (:obj:`bool`): flag on whether or not `SVD DMD` is enabled.

    Attributes:

        LRAN_T (:obj:`int`): the look forward time window length.

        act_fun (:obj:`function`): the activation function chosen.

        sess (:obj:`class`): the session we used in tensorflow.

        prefix (:obj:`str`): the prefix used in the weights and biases parameters, either ``dec`` or ``enc``.

        mode (:obj:`str`): the type of ADVI to use, either ``MAPSimple`` or ``ADVInoARD``, ``ADVIARD``.

        layer_structure_list (:obj:`list`): the layer structure we used in constructing the Feedforward neural network.

        number_layer (:obj:`int`): the number of layers in the network.

        graph (:obj:`class`): the graph assoicated with the session.

        SVD_DMD (:obj:`bool`): flag on whether or not `SVD DMD` is enabled.

        x (:obj:`tf.Tensor`): the placeholder for the input of FNN.

        w_ph_list (:obj:`list`): a list containing the placeholders for weights. Since we treat all NN parameters
            as placeholder, we naturally have a model to use and a NN constructed.

        b_ph_list (:obj:`list`): a list containing the placeohlder for biases.

        output (:obj:`tf.Tensor`): the tensor of the output of the FNN.

        rv_w_list (:obj:`list`): a list of all probability distributions of weights in each layer.

        rv_b_list (:obj:`list`): a list of all probability distributions of biases in each layer.

        noise_lin (:obj:`tf.Tensor`): distribution of linear dynamics noises

        noise_rec (:obj:`tf.Tensor`): distribution of reconstruction dynamics noises

        K_K_X (:obj:`tf.Tensor`): the distribution of K_K_X (it is a normal with softplus scale)

        K_SD (:obj:`tf.Tensor`): the distribution of K_SD (it is log normal)

        num_samples (:obj:`int`): the total number of samples for Monte carlo sampling.

        sample_rv_w_list (:obj:`list`): list of samples of weights in each layer

        sample_rv_b_list (:obj:`list`): list of samples of biases in each layer

        sample_K (:obj:`numpy.ndarray`): array of realization of K matrix.

        sample_lambda_lin (:obj:`numpy.ndarray`): array of realization of scale of linear dynamics noises.

        sample_lambda_rec (:obj:`numpy.ndarray`): array of realization of scale of reconstruction noises.

        sample_noise_lin (:obj:`numpy.ndarray`): array of realization of linear dynamics noises.

        sample_noise_rec (:obj:`numpy.ndarray`): array of realization of reconstruction noises.

    """
    def __init__(self, sess, layer_structure_list, act_fun_str,
                 prefix, mode, LRAN_T, SVD_DMD, POD_W, nsd, W_b_K_dict):

        if LRAN_T != None:

            self.LRAN_T = 1

        else:

            self.LRAN_T = LRAN_T

        # -- choose the activation function
        if act_fun_str == 'tanh':
            self.act_fun = tf.nn.tanh

        elif act_fun_str == 'ptanh':
            self.act_fun = penalized_tanh

        elif act_fun_str == 'elu':
            self.act_fun = tf.nn.elu

        elif act_fun_str == 'swish':
            self.act_fun = myswish_beta

        elif act_fun_str == 'relu':
            self.act_fun = tf.nn.relu

        elif act_fun_str == 'selu':
            self.act_fun = tf.nn.selu

        elif act_fun_str == 'sin':
            self.act_fun = tf.sin

        else:
            raise NotImplementedError('activation function not implemented!')

        self.sess = sess
        self.prefix = prefix
        self.mode = mode
        self.layer_structure_list = layer_structure_list
        self.number_layer = len(self.layer_structure_list)
        self.graph = sess.graph
        self.SVD_DMD = SVD_DMD
        self.POD_W = POD_W
        self.nsd = nsd
        self.W_b_K_dict = W_b_K_dict

        ## create a tensorflow FNN graph: such that weight, bias, K + together with x as input

        with self.graph.as_default():

            # construct placeholder for x
            # Note: such x would be eta for encoder,
            # but it would be phi for decoder. I just use `x` for notation
            self.x = tf.placeholder(TF_FLOAT,
                                    shape=(None, self.layer_structure_list[0]),
                                    name='input_of_FNN')

            # construct a list of placeholder for input of parameters, such that one can evaluate
            self.w_ph_list = []
            self.b_ph_list = []

            for i in xrange(self.number_layer - 1):

                weight_shape = (layer_structure_list[i], layer_structure_list[i + 1])
                bias_shape = (layer_structure_list[i + 1])
                self.w_ph_list.append(tf.placeholder(TF_FLOAT,
                                                     shape=weight_shape,
                                                     name='W_' + str(i + 1)))

                # construct the network with that special structure
                if self.SVD_DMD:
                    if self.prefix == 'enc':
                        if i < self.number_layer - 2:
                            self.b_ph_list.append(tf.placeholder(TF_FLOAT,
                                                                 shape=bias_shape,
                                                                 name='b_' + str(i + 1)))

                    if self.prefix == 'dec':
                        if i > 0:
                            self.b_ph_list.append(tf.placeholder(TF_FLOAT,
                                                                 shape=bias_shape,
                                                                 name='b_' + str(i + 1)))
                else:
                # if SVD DMD is not enabled, we don't do that
                    self.b_ph_list.append(tf.placeholder(TF_FLOAT,
                                                         shape=bias_shape,
                                                         name='b_' + str(i + 1)))


            self.output = None

            # since our likelihood is a diagonal Gaussian, the following
            # is needed in order to propagate the noise from the likelihood

            # so we build a unit Gaussian for multiple variables
            if self.LRAN_T == 1:
                self.unit_GS_rec = MultivariateNormalDiag(
                    loc=tf.zeros(self.layer_structure_list[0], dtype=TF_FLOAT),
                    scale_diag=tf.ones(self.layer_structure_list[0], dtype=TF_FLOAT))

                self.unit_GS_lin = MultivariateNormalDiag(
                    loc=tf.zeros(self.layer_structure_list[-1], dtype=TF_FLOAT),
                    scale_diag=tf.ones(self.layer_structure_list[-1], dtype=TF_FLOAT))
            else:
                self.unit_GS_rec = MultivariateNormalDiag(
                    loc=tf.zeros(self.layer_structure_list[0] * self.LRAN_T, dtype=TF_FLOAT),
                    scale_diag=tf.ones(self.layer_structure_list[0] * self.LRAN_T, dtype=TF_FLOAT))

                self.unit_GS_lin = MultivariateNormalDiag(
                    loc=tf.zeros(self.layer_structure_list[-1] * (self.LRAN_T - 1), dtype=TF_FLOAT),
                    scale_diag=tf.ones(self.layer_structure_list[-1] * (self.LRAN_T - 1), dtype=TF_FLOAT))

    # def construct_linear_FNN(self):
    #     """Construct the feedforward neural network by constructing the graph. Using :attr:`x` as input,
    #     output is :attr:`output` in a linear way. Note that in the createMLPOutLinear of ClassDLKoopmanCont.py.
    #     There is no special linear decoder that only uses a single linear weight. We simply perform symmetrical weights
    #     but turn the nonlinear activation OFF. Since it makes the coding much easier.
    #     """
    #
    #     # simply linear FNN....
    #     MLP = [self.x]
    #     with self.graph.as_default():
    #         for i in xrange(1, len(self.w_ph_list) + 1):
    #             before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #             MLP.append(before_act)
    #
    #         self.output = MLP[-1]
    #
    #     pass

    # def construct_FNN_decoder_linear(self):
    #     """Construct the feedforward neural network by constructing the graph. Using :attr:`x` as input,
    #     output is :attr:`output` in a linear way. Note that in the createMLPOutLinear of ClassDLKoopmanCont.py.
    #     There is no special linear decoder that only uses a single linear weight. We simply perform symmetrical weights
    #     but turn the nonlinear activation OFF. Since it makes the coding much easier.
    #     """
    #
    #     # simply linear FNN....
    #
    #     MLP = [self.x]
    #
    #     with self.graph.as_default():
    #         for i in xrange(1, len(self.w_ph_list) + 1):
    #
    #             if i == len(self.w_ph_list):
    #                 # if it was last layer
    #                 before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #                 MLP.append(before_act)
    #                 # print 'final layer!'
    #             else:
    #
    #                 # other layers
    #                 if self.SVD_DMD and i == 1:
    #                     before_act = tf.matmul(MLP[i - 1], self.w_ph_list[i - 1])
    #                 else:
    #                     before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #
    #                 MLP.append(before_act)
    #
    #         self.output = MLP[-1]

    # def construct_FNN_decoder(self):
    #     """Construct the feedforward neural network by constructing the graph. Using :attr:`x` as input,
    #     output is :attr:`output`.
    #     """
    #
    #     MLP = [self.x]
    #
    #     with self.graph.as_default():
    #         for i in xrange(1, len(self.w_ph_list) + 1):
    #
    #             if i == len(self.w_ph_list):
    #                 # if it was last layer
    #                 before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #                 MLP.append(before_act)
    #                 # print 'final layer!'
    #             else:
    #
    #                 # other layers
    #                 if self.SVD_DMD and i == 1:
    #                     before_act = tf.matmul(MLP[i - 1], self.w_ph_list[i - 1])
    #                 else:
    #                     before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #
    #                 act_fun = self.act_fun
    #                 MLP.append(act_fun(before_act))
    #
    #         self.output = MLP[-1]


    def createMLPOutLinearOLD(self, start_layer, weights,
                           bias, prefix):
        with self.graph.as_default():
            MLP = [start_layer]
            numWeightedLayers = len(weights)

            start_index = 1
            end_index = numWeightedLayers + 1

            # by doing this, we cna have the desired FNN we want, with desired weights and biases
            for indexLayer in xrange(start_index, end_index):
                before_act = tf.add(tf.matmul(MLP[indexLayer - 1],
                                              weights[indexLayer - 1]),
                                    bias[indexLayer - 1])

                if indexLayer < numWeightedLayers:
                    output = self.act_fun(before_act)
                else:
                    output = before_act

                MLP.append(output)

        return MLP[-1]


    def createMLPOutLinear(self, start_layer, weights,
                           bias, prefix):

        with self.graph.as_default():
            MLP = [start_layer]
            numWeightedLayers = len(weights)

            if prefix == 'enc':
                start_index = 1
                end_index = numWeightedLayers

                # by doing this, we cna have the desired FNN we want, with desired weights and biases
                for indexLayer in xrange(start_index, end_index):
                    print('reconstruct encoder NN part...', indexLayer)
                    print('weight =', weights[indexLayer - 1])
                    print('bias =', bias[indexLayer - 1])

                    before_act = tf.add(tf.matmul(MLP[indexLayer - 1],
                                                  weights[indexLayer - 1]),
                                        bias[indexLayer - 1])

                    if indexLayer < (end_index - 1):
                        output = self.act_fun(before_act)
                    else:
                        output = before_act

                    MLP.append(output)

            elif prefix == 'dec':
                start_index = 1
                end_index = numWeightedLayers + 1

                # by doing this, we cna have the desired FNN we want, with desired weights and biases
                for indexLayer in xrange(start_index, end_index):
                    print('reconstruct decoder NN part...', indexLayer)
                    print('weight =', weights[indexLayer - 1])
                    print('bias =', bias[indexLayer - 1])
                    before_act = tf.add(tf.matmul(MLP[indexLayer - 1],
                                                  weights[indexLayer - 1]),
                                        bias[indexLayer - 1])

                    if indexLayer < (end_index - 1):
                        output = self.act_fun(before_act)
                    else:
                        output = before_act

                    MLP.append(output)

            else:
                raise NotImplementedError('not implemented! check prefix!')





        return MLP[-1]


    def construct_FNN(self, prefix):
        """Construct the special feedforward neural network by constructing the graph. Using :attr:`x` as input,
        output is :attr:`output`.
        """
        with self.graph.as_default():

            if prefix == 'enc':

                input = self.x
                # numWeightedLayers = len(self.w_ph_list)

                if self.SVD_DMD:

                    output_NN = self.createMLPOutLinear(start_layer=input,
                                                        weights=self.w_ph_list,
                                                        bias=self.b_ph_list,
                                                        prefix='enc')
                    output_svd = tf.matmul(input, self.POD_W)
                    phi_before_wm = output_NN + output_svd
                    # get last output with weight muplication only, no activation
                    phi = tf.matmul(phi_before_wm, self.w_ph_list[-1])

                else:

                    output_NN = self.createMLPOutLinearOLD(start_layer=input,
                                                        weights=self.w_ph_list,
                                                        bias=self.b_ph_list,
                                                        prefix='enc')
                    phi_before_wm = output_NN
                    phi = phi_before_wm

                output = phi

            elif prefix == 'dec':

                phi = self.x


                # third, depending on if SVD shortcut is enabled or not
                if self.SVD_DMD:
                    # first, take phi with a pure weight multiplication
                    phi_after_wm = tf.matmul(phi, self.w_ph_list[0])

                    # second, take that into a standard Feedforward NN
                    psi_phi_NN = self.createMLPOutLinear(start_layer=phi_after_wm,
                                                         weights=self.w_ph_list[1:],
                                                         bias=self.b_ph_list,
                                                         prefix='dec')

                    psi_svd = tf.matmul(phi_after_wm, self.POD_W)
                    eta_rec = psi_phi_NN + psi_svd
                else:
                    phi_after_wm = phi
                    psi_phi_NN = self.createMLPOutLinearOLD(start_layer=phi_after_wm,
                                                         weights=self.w_ph_list,
                                                         bias=self.b_ph_list,
                                                         prefix='dec')
                    eta_rec = psi_phi_NN

                output = eta_rec

            else:

                raise NotImplementedError('not implemented that prefix = ', prefix)

            self.output = output

        return


        # MLP = [self.x]
        #
        # with self.graph.as_default():
        #     for i in xrange(1, len(self.w_ph_list) + 1):
        #
        #         if i == len(self.w_ph_list):
        #             # if it was last layer
        #             if self.SVD_DMD:
        #                 before_act = tf.matmul(MLP[i - 1], self.w_ph_list[i - 1])
        #             else:
        #                 before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
        #             MLP.append(before_act)
        #             # print 'final layer!'
        #         else:
        #
        #             # print 'i = ', i
        #             before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
        #             act_fun = self.act_fun
        #             MLP.append(act_fun(before_act))
        #
        #     self.output = MLP[-1]

    # def construct_FNN(self):
    #     """Construct the feedforward neural network by constructing the graph. Using :attr:`x` as input,
    #     output is :attr:`output`.
    #     """
    #
    #
    #     MLP = [self.x]
    #
    #     with self.graph.as_default():
    #         for i in xrange(1, len(self.w_ph_list) + 1):
    #
    #             if i == len(self.w_ph_list):
    #                 # if it was last layer
    #                 if self.SVD_DMD:
    #                     before_act = tf.matmul(tf.add(MLP[i - 1], phi_dmd), self.w_ph_list[i - 1])
    #                 else:
    #                     before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #                 MLP.append(before_act)
    #                 # print 'final layer!'
    #             else:
    #
    #                 # print 'i = ', i
    #                 before_act = tf.add(tf.matmul(MLP[i - 1], self.w_ph_list[i - 1]), self.b_ph_list[i - 1])
    #                 act_fun = self.act_fun
    #                 MLP.append(act_fun(before_act))
    #
    #         self.output = MLP[-1]


    def predict(self, x_data, w_list, b_list):
        """Given input data ``x_data``, we output the predictions and given the weights and biases from the
        ``w_list`` and ``b_list``.

        Args:
            x_data (:obj:`numpy.ndarray`): input data

            w_list (:obj:`list`): list of weights for each layer

            b_list (:obj:`list`): list of biases for each layer

        Returns:
            :obj:`numpy.ndarray` : the resulting output.

        """
        with self.graph.as_default():
            feed_dict = {self.x: x_data}

            for i in xrange(self.number_layer - 1):
                feed_dict[self.w_ph_list[i]] = w_list[i]

            if self.SVD_DMD:
                # note that the number of bias is always smaller than the number of weight
                for i in xrange(self.number_layer - 2):
                    feed_dict[self.b_ph_list[i]] = b_list[i].flatten()
            else:
                for i in xrange(self.number_layer - 1):
                    feed_dict[self.b_ph_list[i]] = b_list[i].flatten()
            result = self.sess.run(self.output, feed_dict=feed_dict)

        # the output is either phi or eta_rec
        return result


    def construct_RV(self):
        """Construct the distribution of weights and biases and K, and noises."""

        # obtaining the method that can easily call any tensor by its name

        with self.graph.as_default():

            get_tensor_by_name = self.sess.graph.get_tensor_by_name

            # getting the VI parameters from graph
            # # -- w and b
            # self.rv_w_list = []
            # self.rv_b_list = []




            # sampling noises from Normal distributions
            if self.mode == 'MAP':
                # -- noise
                ones_lin_noise = tf.zeros((self.layer_structure_list[-1]),dtype=TF_FLOAT)
                ones_rec_noise = tf.zeros((self.layer_structure_list[0]),dtype=TF_FLOAT)
                self.noise_lin = Normal(loc=ones_lin_noise, scale=ones_lin_noise)
                self.noise_rec = Normal(loc=ones_rec_noise, scale=ones_rec_noise)

                # -- K
                # self.K = Normal(loc=get_tensor_by_name('qK/loc:0'), scale=tf.zeros(tf.shape(get_tensor_by_name('qK/loc:0'))))

            elif self.mode == 'ADVIARD':

                # the following assumes that the noise VP is log-normal.
                # so we need ``ed.models.TransformedDistribution``

                # -- noise, the variance of the noise
                self.noise_lin = ed.models.TransformedDistribution(
                    distribution=ed.models.NormalWithSoftplusScale(
                        loc=get_tensor_by_name('scale_noise_lin/loc:0'),
                        scale=get_tensor_by_name('scale_noise_lin/scale:0')),
                    bijector=bijectors.Exp(),
                    name='noise_lin')

                self.noise_rec = ed.models.TransformedDistribution(
                    distribution=ed.models.NormalWithSoftplusScale(
                        loc=get_tensor_by_name('scale_noise_rec/loc:0'),
                        scale=get_tensor_by_name('scale_noise_rec/scale:0')),
                    bijector=bijectors.Exp(),
                    name='noise_rec')

                # -- K
                # self.K = Normal(loc=get_tensor_by_name('qK/loc:0'), scale=tf.nn.softplus(get_tensor_by_name('qK/scale:0')))

            elif self.mode == 'ADVInoARD':

                ADVInoARD_noise_scale = tf.constant(1e-3, dtype=TF_FLOAT)

                # -- noise
                ones_lin_noise = tf.ones(self.layer_structure_list[-1],dtype=TF_FLOAT)
                ones_rec_noise = tf.ones(self.layer_structure_list[0],dtype=TF_FLOAT)

                # this would just be the constant, since in ADVInoARD, the noise in likelihood is fixed.....
                self.noise_lin = Normal(loc=ones_lin_noise * ADVInoARD_noise_scale, scale=tf.zeros(ones_lin_noise.shape, dtype=TF_FLOAT) )
                self.noise_rec = Normal(loc=ones_rec_noise * ADVInoARD_noise_scale, scale=tf.zeros(ones_rec_noise.shape, dtype=TF_FLOAT) )

                # -- K
                # self.K = Normal(loc=get_tensor_by_name('qK/loc:0'), scale=tf.nn.softplus(get_tensor_by_name('qK/scale:0')))

            else:

                raise NotImplementedError('Error noise cannot be sampled!!')

        return


    def sampling_RV(self, num_samples):
        """ sampling Random Variables from their distributions to get ``num_samples`` realizations.

        Args:

            num_samples (:obj:`int`): number of Monte Carlo samples.

        """

        with self.graph.as_default():

            # set number of samples
            self.num_samples = num_samples

            # sampling noise
            if self.prefix == 'enc':
                self.sample_noise_lin = self.sess.run(self.unit_GS_lin.sample(num_samples)
                                                      * tf.sqrt(self.noise_lin.sample(num_samples)))
                self.sample_noise_rec = self.sess.run(self.unit_GS_rec.sample(num_samples)
                                                      * tf.sqrt(self.noise_rec.sample(num_samples)))
                self.sample_lambda_lin = self.sess.run(self.noise_lin.sample(num_samples))
                self.sample_lambda_rec = self.sess.run(self.noise_rec.sample(num_samples))


            # sampling w and b
            self.sample_rv_w_list = []
            self.sample_rv_b_list = []

            # read weights and biases and K from ``weight_bias_save.npy``
            print self.W_b_K_dict.keys()

            if self.prefix == 'enc':
                for i in range(1, len(self.layer_structure_list)):
                    self.sample_rv_w_list.append(self.W_b_K_dict['enc_w_' + str(i)])
                    if self.SVD_DMD:
                        if i < len(self.layer_structure_list) - 1:
                            self.sample_rv_b_list.append(self.W_b_K_dict['enc_b_'+str(i)])
                    else:
                        self.sample_rv_b_list.append(self.W_b_K_dict['enc_b_' + str(i)])

            elif self.prefix == 'dec':
                for i in range(1, len(self.layer_structure_list)):
                    self.sample_rv_w_list.append(self.W_b_K_dict['dec_w_' + str(i)])
                    if self.SVD_DMD:
                        if i > 1:
                            self.sample_rv_b_list.append(self.W_b_K_dict['dec_b_'+ str(i)])
                    else:
                        self.sample_rv_b_list.append(self.W_b_K_dict['dec_b_' + str(i)])
            else:

                raise NotImplementedError('not implemented!')

            # read K from ``weight_bias_save.npy``
            self.sample_K = self.W_b_K_dict['K']

        #
        # for index in xrange(len(self.rv_w_list)):
        #     self.sample_rv_w_list.append(self.sess.run(self.rv_w_list[index].sample(num_samples)))
        #     self.sample_rv_b_list.append(self.sess.run(self.rv_b_list[index].sample(num_samples)))



        # sampling K from nsd==1 version
        # samples_X_upper = self.sess.run(tf.matrix_band_part(self.K_K_X.sample(num_samples), 0, 1))
        #
        # samples_K_SD = self.sess.run(self.K_SD.sample(num_samples))
        #
        # samples_K_SD_list = []
        # for index in xrange(samples_K_SD.shape[0]):
        #     samples_K_SD_list.append(np.diag(samples_K_SD[index]))
        # samples_K_SD = np.array(samples_K_SD_list)
        #
        # samples_X_upper_T = np.einsum('ijk->ikj', samples_X_upper)
        #
        # samples_K_XX = samples_X_upper - samples_X_upper_T
        #
        # samples_K = samples_K_XX - samples_K_SD
        #
        # self.sample_K = samples_K

    #######################################

    def get_K(self):
        return self.sample_K

    def get_lambda_lin(self):
        return self.sample_lambda_lin

    def get_lambda_rec(self):
        return self.sample_lambda_rec

    def get_noise_lin(self):
        return self.sample_noise_lin

    def get_noise_rec(self):
        return self.sample_noise_rec

    def ensemble_predict(self, x_data):
        """output the ensemble of the predictions, of all Monte Carlo samplings

        Args:
            x_data (:obj:`numpy.ndarray`): the initial condition, i.e., input data.

        Returns:
            :obj:`list` : a list of all predictions for all Monte Carlo samples.

        """
        # noise_list = self.sess.run(self.sample_noise_lin)

        pred_list = []

        for i in xrange(self.num_samples):
            w_list = []
            b_list = []
            for j in xrange(len(self.sample_rv_w_list) ):
                w_list.append(self.sample_rv_w_list[j][i,:,:])

            if self.SVD_DMD:
                for j in xrange(len(self.sample_rv_w_list)-1):
                    b_list.append(self.sample_rv_b_list[j][i,:,:])
            else:
                for j in xrange(len(self.sample_rv_w_list)):
                    b_list.append(self.sample_rv_b_list[j][i,:,:])

            pred_list.append(self.predict(x_data=x_data,
                                          w_list=w_list,
                                          b_list=b_list) )

        return pred_list

    def predict_with_i_realization(self, x_data, i):
        """predicting with certain realizations

        Args:
            x_data (:obj:`numpy.ndarray`): the initial condition, i.e., input data.

            i (:obj:`int`): the index of a certain Monte carlo samples.

        Returns:
            :obj:`numpy.ndarray` : the array of output for that given realization.

        """

        w_list = []
        b_list = []

        for j in xrange(len(self.sample_rv_w_list)):
            w_list.append(self.sample_rv_w_list[j][i,:,:])

        if self.SVD_DMD:
            for j in xrange(len(self.sample_rv_w_list)-1):
                b_list.append(self.sample_rv_b_list[j][i,:,:])
        else:
            for j in xrange(len(self.sample_rv_w_list)):
                b_list.append(self.sample_rv_b_list[j][i,:,:])


        return self.predict(x_data=x_data.reshape(1,-1), w_list=w_list, b_list=b_list)


    def get_dependencies(self, tensor):
        dependencies = set()
        dependencies.update(tensor.op.inputs)
        for sub_op in tensor.op.inputs:
            dependencies.update(self.get_dependencies(sub_op))
        return dependencies

    def get_placeholder_of_tensor(self, tensor):
        return [tensor for tensor in self.get_dependencies(tensor) if tensor.op.type == 'Placeholder']


class ClassModelBayesDLKoopman(object):
    """Class of Bayesian Model Interface with Deep Learning Koopman

    Args:
        model_path (:obj:`str`): path to the Bayesian model.

        n_mc_samples (:obj:`int`): the number of Monte carlo samplings.

        mode (:obj:`str`): the string for the mode.

        n_mc_diff_samples (:obj:`int`): the number of MC samples for differential form, i.e., for that MVOU process part.

        gpu_percentage (:obj:`float`): GPU memory ratio.

    Attributes:
        sess (:obj:`class`): current session we are in using Tensorflow with GPU configured.

        graph (:obj:`class`): the graph.

        encoder_layer_structure (:obj:`list`): a list containing the encoder structures.

        act_fun (:obj:`str`): the string for activation functions

        encoder_POD (:obj:`numpy.ndarray`): POD weights coefficients.

        decoder_POD (:obj:`numpy.ndarray`): POD weights coefficients.

        SVD_DMD (:obj:`bool`): enable SVD_DMD or not.

        nonlinear_rec (:obj:`numpy.ndarray`): enable nonlinear reconstruction or not. Default is True.

        n_mc_samples (:obj:`int`): the number of Monte carlo samplings.

        n_mc_diff_samples (:obj:`int`): the number of MC samples for differential form, i.e., for that MVOU process part.

        LRAN_T (:obj:`int`): future prediction window length. ``1`` means differential form.

        encoder_fnn_edward (:obj:`class`): encoder FNN input-output model.

        decoder_fnn_edward (:obj:`class`): encoder FNN input-output model.

        DMD_K (:obj:`numpy.ndarray`): previous I used this to trully embedded DMD into it. But I didn't continue because DMD can be unstable.

        mu_X (:obj:`numpy.ndarray`): normalization parameters :math:`\overline{X}`

        Lambda_X (:obj:`numpy.ndarray`): normalization parameters :math:`\Lambda`

        Lambda_X_inv (:obj:`numpy.ndarray`): normalization parameters :math:`\Lambda^{-1}`

    """

    def __init__(self, model_path, n_mc_samples=500, mode=None,
                 n_mc_diff_samples=10, gpu_percent=0.2):

        # we setup the GPUs sessions, graphs.
        try:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_percent)
            self.sess = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
        except:
            self.sess = tf.Session(graph=tf.Graph())
        _ = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_path)
        self.graph = self.sess.graph

        # 1. we read meta data from `model_arch.npz`
        data_dict = dict(np.load(model_path + '/model_arch.npz'))
        self.encoder_layer_structure = data_dict['encoder_layer_structure']
        self.act_fun = data_dict['act_fun']
        self.encoder_POD = data_dict['encoder_POD_weights']
        self.decoder_POD = data_dict['decoder_POD_weights']
        self.SVD_DMD = data_dict['SVD_DMD'] == True
        self.nonlinear_rec = data_dict['nonlinear_rec']
        self.nsd = data_dict['nsd']

        print('')
        print 'enable nonlinear reconstruction? ', self.nonlinear_rec
        print 'enable SVD_DMD? ', self.SVD_DMD
        print 'encoder POD weights = \n', self.encoder_POD
        print 'decoder POD weights = \n', self.decoder_POD

        # number of monte carlo samples
        self.n_mc_samples = n_mc_samples

        # number of monte carlo samples for diff Mu process
        # note: this is only used when you run in ``differential form``
        self.n_mc_diff_samples = n_mc_diff_samples

        # 2. read encoder and decoder neural network W and b samplings
        W_b_K_dict = np.load(model_path + '/../weight_bias_save.npy')
        W_b_K_dict = W_b_K_dict.item()



        # detect if the
        try:
            self.LRAN_T = self.graph.get_tensor_by_name('scale_noise_rec/loc:0').shape[0] / self.encoder_layer_structure[0]
            print('getting Recurrent model with T = ', self.LRAN_T)
        except:
            self.LRAN_T = 1
            print('getting Differential model')


        # build encoder, as a edward fnn model
        self.encoder_fnn_edward = ClassFNN_edward(sess=self.sess,
                                                  layer_structure_list=self.encoder_layer_structure,
                                                  act_fun_str=self.act_fun,
                                                  prefix='enc',
                                                  mode=mode,
                                                  LRAN_T=self.LRAN_T,
                                                  SVD_DMD=self.SVD_DMD,
                                                  POD_W = self.encoder_POD,
                                                  nsd=self.nsd,
                                                  W_b_K_dict=W_b_K_dict)

        # construct the encoder.
        # note that SVD DMD is inside this function already
        self.encoder_fnn_edward.construct_FNN(prefix='enc')

        # construct the random variables
        self.encoder_fnn_edward.construct_RV()
        self.encoder_fnn_edward.sampling_RV(num_samples=n_mc_samples)

        # build decoder, as a edward fnn model

        if self.nonlinear_rec:

            decoder_layer_structure = self.encoder_layer_structure[::-1]
        else:
            if self.SVD_DMD:
                decoder_layer_structure = [self.encoder_layer_structure[-1],
                                           self.encoder_layer_structure[-2],
                                           self.encoder_layer_structure[0]]
            else:
                decoder_layer_structure = [self.encoder_layer_structure[-1],
                                           self.encoder_layer_structure[0]]

        self.decoder_fnn_edward = ClassFNN_edward(sess=self.sess,
                                                  layer_structure_list=decoder_layer_structure,
                                                  act_fun_str=self.act_fun,
                                                  prefix='dec',
                                                  mode=mode,
                                                  LRAN_T=self.LRAN_T,
                                                  SVD_DMD=self.SVD_DMD,
                                                  POD_W=self.decoder_POD,
                                                  nsd=self.nsd,
                                                  W_b_K_dict=W_b_K_dict)

        # construct the decoder
        self.decoder_fnn_edward.construct_FNN(prefix='dec')
        # if self.nonlinear_rec:
        #     self.decoder_fnn_edward.construct_FNN_decoder()
        # else:
        #     self.decoder_fnn_edward.construct_FNN_decoder_linear()

        # construct the random variables

        self.decoder_fnn_edward.construct_RV()
        self.decoder_fnn_edward.sampling_RV(num_samples=n_mc_samples)


        self.DMD_K = 0

        # obtain Normalization stuff..
        # note that in the above, we don't have any normalization applied at all..

        self.mu_X = self.sess.run(self.graph.get_tensor_by_name('mu_X:0'))
        self.Lambda_X = self.sess.run(self.graph.get_tensor_by_name('Lambda_X:0'))
        self.Lambda_X_inv = np.linalg.inv(self.Lambda_X)

    def get_K(self):
        """get realization of K

        Returns:

            :obj:`numpy.ndarray` : realization of K

        """
        MC_K = self.encoder_fnn_edward.get_K()
        return MC_K

    def transform_x_to_eta(self, x):
        """transformation from :math:`x` to :math:`\eta`

        Args:

            x (:obj:`numpy.ndarray`): physical states :math:`x`

        Returns:

            :obj:`numpy.ndarray` : normalized states :math:`\eta`

        """
        return np.matmul(x - self.mu_X, self.Lambda_X_inv)

    def transform_eta_to_x(self, eta):
        """transformation from :math:`\eta` to :math:`x`

        Args:

            eta (:obj:`numpy.ndarray`) : normalized states :math:`\eta`

        Returns:

            :obj:`numpy.ndarray` : physical states :math:`x`

        """

        return self.mu_X + np.matmul(eta, self.Lambda_X)

    def get_lambda_lin(self):
        return self.encoder_fnn_edward.get_lambda_lin()

    def get_lambda_rec(self):
        return self.encoder_fnn_edward.get_lambda_rec()

    def get_noise_lin(self):
        return self.encoder_fnn_edward.get_noise_lin()

    def get_noise_rec(self):
        return self.encoder_fnn_edward.get_noise_rec()

    # def compute_phi_nn(self, x):
    #     """compute the :math:`\phi` of the contribution from neural network.
    #
    #     First we transform :math:`x` into :math:`\phi`, then we feed into neural network.
    #
    #     Args:
    #
    #         x (:obj:`numpy.ndarray`): input data.
    #
    #     Returns:
    #
    #         :obj:`numpy.ndarray` : observables :math:`\phi` of NN contribution.
    #
    #     """
    #
    #     # outputs is a list with Q length, each of them is a (1,K) array
    #     eta = self.transform_x_to_eta(x)
    #     phi_nn_mc_array = np.array(self.encoder_fnn_edward.ensemble_predict(x_data=eta))
    #
    #     return phi_nn_mc_array

    # def compute_phi_dmd(self, x):
    #     """compute the :math:`\phi` of the contribution from DMD.
    #
    #     Args:
    #
    #         x (:obj:`numpy.ndarray`): input data.
    #
    #     Returns:
    #
    #         :obj:`numpy.ndarray` : observables :math:`\phi` of DMD contribution.
    #
    #     """
    #
    #     # outputs is a list with Q length, each of them is a (1,K) array
    #
    #     eta = self.transform_x_to_eta(x)
    #     ## I don't get this why I mulitply lambda_x twice
    #     # phi_dmd_mc_array = np.tile(np.matmul(np.matmul(eta, self.Lambda_X), self.encoder_POD), (self.n_mc_samples, 1, 1))
    #     phi_dmd_mc_array = np.tile(np.matmul(eta, self.encoder_POD), (self.n_mc_samples, 1, 1))
    #
    #     # then, get the distribution of the LAST weight of encoder
    #     W_enc_last_rv = self.encoder_fnn_edward.sample_rv_w_list[-1]
    #
    #     # adding the linear constribution
    #     for i_sample, W_enc_last_rv_sample in enumerate(W_enc_last_rv):
    #         phi_dmd_mc_array[i_sample] = np.matmul(phi_dmd_mc_array[i_sample], W_enc_last_rv_sample)
    #
    #     return phi_dmd_mc_array

    def computePhi(self, x):
        """Compute the :math:`\phi` given :math:`x`

        Args:

            x (:obj:`numpy.ndarray`): given state :math:`x`.

        Returns:

            :obj:`numpy.ndarray` : observables :math:`\phi`.

        """

        # input an array with size (1,M) of x, which is the initial state as M is the degree of freedom for the system
        # then we output a size of matrix of (Q,K), where K is the size of Koopman dimension
        # Q is number of realizations

        eta = self.transform_x_to_eta(x)
        Phi = np.array(self.encoder_fnn_edward.ensemble_predict(x_data=eta))


        # if self.SVD_DMD:
        #     # TRUE PHI = each row of QK_array + np.matmul(eta, self.encoder_POD)
        #     true_Phi = self.compute_phi_dmd(x) + self.compute_phi_nn(x)
        # else:
        #     true_Phi = self.compute_phi_nn(x)

        return Phi

    def computeEigenPhi(self, x):
        pass

    # def reconstruct_with_i_realization_nn(self, phi_nn, i):
    #     i_realization_M_array = self.decoder_fnn_edward.predict_with_i_realization(x_data=phi_nn, i=i)
    #     return i_realization_M_array
    #
    # def reconstruct_with_i_realization_dmd(self, phi_dmd):
    #     return np.matmul(phi_dmd, np.matmul(self.decoder_POD, self.Lambda_X_inv))
    #
    # def reconstruct_with_i_realization_svd_dmd(self, phi_nn, phi_dmd, i):
    #     eta_rec = self.reconstruct_with_i_realization_dmd(phi_dmd=phi_dmd) + self.reconstruct_with_i_realization_nn(phi_nn=phi_nn,i=i)
    #     return eta_rec

    def reconstruct_with_i_realization(self, phi, i):
        """Compute reconstructed state :math:`x` from :math:`\phi` for certain realization.

        Args:

            phi (:obj:`numpy.ndarray`): the Koopman observables.

            i (:obj:`int`): index for realization.

        Returns:

            :obj:`numpy.ndarray` : states reconstructed.

        """
        # input Phi is a i-th (1,K) Phi(t)

        # output is the i-th realization of Phi^{-1}, on this input Phi, which i-th eta_rec
        i_realization_M_array = self.decoder_fnn_edward.predict_with_i_realization(x_data=phi, i=i)

        # if self.SVD_DMD:
        #     # TRUE States = QM_array + np.matmul(Phi, self.decoder_POD)
        #
        #     # multiply with the i-th realization of the first weight of decoder
        #     phi_after_first_decoder = np.matmul(phi, self.decoder_fnn_edward.sample_rv_w_list[0][i])
        #
        #     ## again, I don't get this why I mulitply lambda_x twice
        #     # linear_reconstruction = np.matmul(phi_after_first_decoder, np.matmul(self.decoder_POD, self.Lambda_X_inv))
        #     linear_reconstruction = np.matmul(phi_after_first_decoder, self.decoder_POD)
        #
        #     true_states = i_realization_M_array + linear_reconstruction
        # else:
        #     true_states = i_realization_M_array

        eta_rec = i_realization_M_array

        return eta_rec

    def close(self):
        self.sess.close()


class ClassModelDLKoopmanLRAN(object):
    """Class for Model interface with deterministic LRAN

    Args:

        model_path (:obj:`str`): the path of the model

    Attributes:

        sess (:obj:`class`): the session we used in tensorflow.

        graph (:obj:`class`): the graph assoicated with the session.

        x (:obj:`tf.Tensor`): input placeholder states

        future (:obj:`tf.Tensor`): output tensor for future step prediction

        linearEvolving (:obj:`tf.Tensor`): the tensor of the K matrix.

        phi (:obj:`tf.Tensor`): the tensor of the Koopman observables

        xrec (:obj:`tf.Tensor`): the tensor of the recontructed state

    """
    def __init__(self, model_path):

        # initialize the session

        try:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
            self.sess = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
        except:
            self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_path)

        self.graph = self.sess.graph

        # get necessary tensor from the existing graph
        self.x = self.graph.get_tensor_by_name("X0:0")
        self.future = self.graph.get_tensor_by_name("Xfuture:0")

        self.linearEvolving = self.sess.run('K:0')
        self.phi = self.graph.get_tensor_by_name("phi:0")
        self.xrec = self.graph.get_tensor_by_name('Xrec:0')

    def computePhi(self, x):
        """ compute :math:`\phi` given :math:`x`.

        Args:

            x (:obj:`numpy.ndarray`): the input states

        Returns:

            :obj:`numpy.ndarray` : observables.

        """
        feed_dict = {self.x: x}
        return self.sess.run([self.phi], feed_dict)[0]

    def get_linearEvolving(self):
        """
        Get K matrix.

        Returns:

            :obj:`numpy.ndarray` : Koopman matrix ``K``.

        """
        return self.linearEvolving

    def reconstruct(self, Phi):
        """
        Get reconstruction from :math:`\phi`

        Returns:

            :obj:`numpy.ndarray` : reconstruction.

        """
        feed_dict = {self.phi: Phi}
        return self.sess.run([self.xrec], feed_dict)[0]

    def close(self):
        self.sess.close()


class ClassModelDLKoopman(object):
    """Class of Deep learning Koopman model

    Args:
        model_path (:obj:`str`): path for the model that is saved in a simple save from Tensorflow

    Attributes:
        sess (:obj:`class`): the session we used in tensorflow.

        graph (:obj:`class`): the graph assoicated with the session.

        x (:obj:`tf.Tensor`): input placeholder states :math:`x`

        xdot (:obj:`tf.Tensor`): input placeholder states derivative :math:`\dot{x}`

        linearEvolving (:obj:`numpy.ndarray`): matrix of K

        D (:obj:`numpy.ndarray`): Koopman eigenvalues

        R (:obj:`numpy.ndarray`): Koopman eigenvectors

        xrec (:obj:`tf.Tensor`): the tensor of the reconstructed states

        linearEvolvingEigen (:obj:`numpy.ndarray`): Koopman eigenvalues diagonal matrix.

        phi (:obj:`tf.Tensor`): tensor of the Koopman observables.

    """
    def __init__(self, model_path):

        # initialize the session

        try:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
            self.sess = tf.Session(graph=tf.Graph(),config=tf.ConfigProto(gpu_options=gpu_options))
        except:
            self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_path)

        # weird problem of tensorflow simple load
        # https://stackoverflow.com/questions/43701902/how-to-keep-tensorflow-session-open-between-predictions-loading-from-savedmodel?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        # tf.saved_model.loader.load has some weird interactions between the default graph and the graph that is part of the session
        self.graph = self.sess.graph  # tf.get_default_graph()

        # retrieve feed dict tensor
        self.x = self.graph.get_tensor_by_name("X:0")
        self.xdot = self.graph.get_tensor_by_name("Xdot:0")
        # self.phase = self.graph.get_tensor_by_name("phase:0")

        # retrieve K
        self.linearEvolving = self.sess.run('K:0')  # np.ndarray

        # retrieve eigen values
        [self.D, self.R] = LA.eig(self.linearEvolving)
        self.linearEvolvingEigen = np.diag(self.D)

        # retrieve desired ops
        self.phi = self.graph.get_tensor_by_name("phi:0")
        self.xrec = self.graph.get_tensor_by_name("Xrec:0")

    def get_linearEvolvingEigen(self):
        return self.linearEvolvingEigen

    def get_linearEvolving(self):
        return self.linearEvolving

    def computePhi(self, x):
        """compute nonlinear observables, given state x

        Args:

            x (:obj:`numpy.ndarray`): state, :math:`x`

        Returns:

            :obj:`numpy.ndarray` : phi, :math:`\phi`

        """
        feed_dict = {self.x: x, self.xdot: x}
        return self.sess.run([self.phi], feed_dict)[0]

    def computeEigenPhi(self, x):
        """compute Koopman eigenfunctions given x

        Args:

            x (:obj:`numpy.ndarray`): input state

        Returns:

            :obj:`numpy.ndarray` : output Koopman eigenfunction values at :math:`x`.

        """

        # compute the nonlinear observables
        phi = self.computePhi(x)
        # compute the eigenfunctions value
        phi_eigen = np.matmul(phi, self.R)
        return phi_eigen

    def reconstruct(self, Phi):
        """reconstruct state from nonlinear observable Phi

        Args:

            Phi (:obj:`numpy.ndarray`): Koopman observable

        Returns:

            :obj:`numpy.ndarray` : reconstructed states

        """
        feed_dict = {self.phi: Phi}
        return self.sess.run([self.xrec], feed_dict)[0]

    def close(self):
        """close the session to free the resources
        """
        self.sess.close()


def F_2d_duffing_system_interface(t, y):
    """
    Governing equation for 2D duffing system using Otto 2017 paper: for Python RK4 purpose

    Args:
        t (:obj:`float`): time
        y (:obj:`numpy.ndarray`): the state of the system :math:`x`.

    Returns:
        :obj:`numpy.ndarray` : the time derivative :math:`\dot{x}`
    """

    input = np.array(y)
    return F_duffing_2d_system(input)[0]


def F_simple_2d_system_interface(t, y):
    """
    Governing equation for 2D simple system of Lusch 2017 paper: for Python RK4 purpose

    Args:
        t (:obj:`float`): time
        y (:obj:`numpy.ndarray`): the state of the system :math:`x`.

    Returns:
        :obj:`numpy.ndarray` : the time derivative :math:`\dot{x}`
    """

    input = np.array(y)
    return F_simple_2d_system(input)[0]
