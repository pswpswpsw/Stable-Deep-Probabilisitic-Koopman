#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class of Continuous time Deep Learning Models for Koopman Operator with LRAN"""

import sys

sys.dont_write_bytecode = True
import pprint

from ClassDLKoopmanCont import *


class ClassDLKoopmanLRAN(ClassDLKoopmanCont):
    """Class of LRAN

    Simply using multiple steps in to the future, i.e., :attr:`T`, for prediction and minimize error. Moreover,
    data is split into training and testing by their sequential order.

    Args:
        configDict (:obj:`dict`): dictionary of the configuration on deterministic side for LRAN

        edward_configDict (:obj:`dict`): dictionary of configuration on Bayesian DL model.

        gpu_percentage (:obj:`float`): percentage for GPU memory usage.

    Attributes:
        T (:obj:`int`): number of steps of future to look forward to see.

        dt (:obj:`float`): time interval between snapshots since I am doing LRAN, I need `dt` since the reconstruction
            requires time.

        Xraw (:obj:`numpy.ndarray`): raw trajectory with no hankerization and normalization.

        hankel_data (:obj:`numpy.ndarray`): hankerized data with :attr:`T`.

        future_X_list (:obj:`list`): a list contains all future state output from the neural network.

        future_eta_list (:obj:`list`): a list contains all future state in normalized form from neural network.

        future_koopman_phi_list (:obj:`list`): a list contains all future koopman phi :math:`\phi`.

        future_koopman_phi_evolved_list (:obj:`list`): a list contains all
            future koopman evolved from the neural network.

        future_Xrec_list (:obj:`list`): a list contains reconstructed states from the neural network.

        future_etaRec_list (:obj:`list`): a list contains reconstructed normalized states from the neural network.

        XfutureTrain (:obj:`numpy.ndarray`): array contains the future training states.



    """

    def __init__(self, configDict, edward_configDict=None, gpu_percentage=0.25):

        # call DLKoopmanCont case to initialize

        ClassDLKoopmanCont.__init__(self, configDict, edward_configDict, gpu_percentage)

        # get the number of future state forcasting in LRAN

        self.T = configDict.get('T')

        # get the dt in LRAN

        self.dt = configDict.get('dt')

    def convert_discrete_time_trajectory_into_hankel(self, data_trajectory, T):
        """Convert discrete time trajectory into hankel matrix.

        Args:
            data_trajectory (:obj:`numpy.ndarray`): a single data trajectory
            with first axis as number of snapshots.

        Returns:
            :obj:`numpy.ndarray` : snapshots hankerized.

        """

        n_sample = data_trajectory.shape[0]
        n_dim = data_trajectory.shape[1]

        hankel_data = np.zeros((n_sample - T + 1, T, n_dim))
        for i in xrange(n_sample - T + 1):
            hankel_data[i, :, :] = data_trajectory[i:i + T, :]

        # the returned data is (n_sample - T + 1, T, n_dim)
        return hankel_data

    def get_hankel_data(self, X, valid_size=0.01):
        """ Get hankel data and normalized it, then get training and testing and PODs.

        Args:
            X (:obj:`numpy.ndarray`): a single trajectory.

            valid_size (:obj:`float`): validation ratio in all the data.


        Note:
            - for trajectory data, there is no training and testing, we use all of the trajectory
            - we normalize the trajectory later in the process...

        """

        if X.ndim == 2:
            # get raw data for transforming
            print('single trajectory training')
            Xraw = X
        else:
            # stacking all trajectory data
            print('multiple trajectory training')
            Xraw = np.vstack(X)  # shape = (N_TRJ, N_SAMPLE, N_DIM)

        # get data normalized, find mean and std, std can be found by different ways..
        # -- keep component variance the same
        # -- or not keep the variance among components differently

        scaler_X = StandardScaler()

        # modification to match Seth & Kutz 2019 paper.
        # -- we choose to normalize by subtract the mean partially from training data. 
        # -- I know it is sub-optimal to subtract from equilibirum

        if X.ndim == 2:
            data_for_scaling = Xraw[:-self.T, :]
        else:
            data_for_scaling = np.vstack(Xraw[:, :-self.T, :])

        scaler_X.fit(data_for_scaling)

        self.mu_X = tf.convert_to_tensor(scaler_X.mean_, dtype=TF_FLOAT, name='mu_X')

        if self.model_params['normalization_type'] == 'only_max':
            variance = np.ones(scaler_X.var_.shape) * np.max(scaler_X.var_)
        else:
            variance = scaler_X.var_

        self.Lambda_X = tf.convert_to_tensor(np.diag(np.sqrt(variance)), dtype=TF_FLOAT, name='Lambda_X')
        self.Lambda_X_inv = tf.linalg.inv(self.Lambda_X)  # tf.convert_to_tensor(np.linalg.inv(self.Lambda_X), dtype=tf.float32)

        # arrange trajectory data into training and validation segements
        if X.ndim == 2:

            # get hankel data from the unnormalized data, later in the graph, we normalize for every input data, so we don't normalize anything here
            hankel_data = self.convert_discrete_time_trajectory_into_hankel(data_trajectory=Xraw, T=self.T)
            total_data_number = hankel_data.shape[0]

            Xtrain = hankel_data[:int(total_data_number * (1 - valid_size)), 0, :]  # initial state
            XfutureTrain = hankel_data[:int(total_data_number * (1 - valid_size)), 1:, :]  # the

            Xvalid = hankel_data[int(total_data_number * (1 - valid_size)):, 0, :]
            XfutureValid = hankel_data[int(total_data_number * (1 - valid_size)):, 1:, :]

        else:

            Xtrain_list = []
            XfutureTrain_list = []
            Xvalid_list = []
            XfutureValid_list = []

            for J in range(X.shape[0]):
                # for each J trajectory get hankel data

                hankel_data = self.convert_discrete_time_trajectory_into_hankel(data_trajectory=Xraw[J, :, :], T=self.T)
                total_data_number = hankel_data.shape[0]
                Xtrain_J = hankel_data[:int(total_data_number * (1 - valid_size)), 0, :]
                XfutureTrain_J = hankel_data[:int(total_data_number * (1 - valid_size)), 1:, :]
                Xvalid_J = hankel_data[int(total_data_number * (1 - valid_size)):, 0, :]
                XfutureValid_J = hankel_data[int(total_data_number * (1 - valid_size)):, 1:, :]

                Xtrain_list.append(Xtrain_J)
                XfutureTrain_list.append(XfutureTrain_J)
                Xvalid_list.append(Xvalid_J)
                XfutureValid_list.append(XfutureValid_J)

            Xtrain = np.vstack(Xtrain_list)
            XfutureTrain = np.vstack(XfutureTrain_list)
            Xvalid = np.vstack(Xvalid_list)
            XfutureValid = np.vstack(XfutureValid_list)

        # we split data into training and validation by sequential order.., no random shuffling!
        self.Xtrain = Xtrain
        self.XfutureTrain = XfutureTrain

        self.Xvalid = Xvalid
        self.XfutureValid = XfutureValid

        # compute POD transformation, only for one state, no embedding
        X_train_normalized = scaler_X.transform(self.Xtrain)
        u, s, vh = np.linalg.svd(X_train_normalized, full_matrices=False)
        self.POD_V = tf.convert_to_tensor(vh.transpose(), dtype=TF_FLOAT)

        return

    def construct_model(self):
        """Construct the LRAN model

        Returns:
            :obj:`tuple` : (residual_vector_rec_loss, residual_vector_lin_loss)

        """

        # Setup dimension and activations

        self._set_dimension_and_activations()

        # input the current state

        self.X = tf.placeholder(TF_FLOAT, [None, self.dimX], name='X0')

        # input a list of future X state as placeholders

        self.future_X_list = []
        for tau in xrange(1, self.T):
            self.future_X_list.append(tf.placeholder(TF_FLOAT, [None, self.dimX], name='X' + str(tau)))

        # so there will be placeholder as self.X = X0
        # and self.future_X_list = [X_1,...,X_{T-1}]

        # transform into eta: normalized...

        self.eta = tf.matmul(self.X - self.mu_X, self.Lambda_X_inv)

        # transform each future state into normalized

        self.future_eta_list = []
        for tau in xrange(1, self.T):
            self.future_eta_list.append(tf.matmul(self.future_X_list[tau - 1] - self.mu_X, self.Lambda_X_inv))

        # auxiliary variable, we will feed it in the graph..

        self.z_rec_loss = tf.placeholder(TF_FLOAT, [None, self.T * self.dimX], name='Z_rec_loss')
        self.z_lin_loss = tf.placeholder(TF_FLOAT, [None, (self.T - 1) * self.numKoopmanModes], name='Z_lin_loss')

        # setup hyperprior and prior dictionary

        self.setup_K()

        # Setup encoder & decoder weights

        self._create_encoder_decoder_and_pod_weights()

        print('')
        print('printing the defined priors')
        print(self.prior_dict.keys())
        print('')
        print('printing the defined posterioris')
        print(self.vp_dict.keys())

        # after all weights, bias, K are defined (along with their priors and posterioris)
        # -- > let's build the network topology
        # 1. encoding

        self.koopmanPhi = self.encoding_neural_net_with_pod(input=self.eta,
                                                            SVD_DMD_FLAG=self.model_params['SVD_DMD'])
        self.koopmanPhi = tf.identity(self.koopmanPhi, name='phi')

        # 2. decoding

        self.etaRec = self.decoding_neural_net_with_pod(phi=self.koopmanPhi,
                                                        SVD_DMD_FLAG=self.model_params['SVD_DMD'])

        # 3. scaling etc_rec back to X

        self.Xrec = tf.add(self.mu_X, tf.matmul(self.etaRec, self.Lambda_X), name='Xrec')

        # now for each future state, find the future koopman phi

        self.future_koopman_phi_list = []

        for tau in xrange(1, self.T):
            phi = self.encoding_neural_net_with_pod(input=self.future_eta_list[tau - 1],
                                                    SVD_DMD_FLAG=self.model_params['SVD_DMD'])
            self.future_koopman_phi_list.append(phi)

        # now time evolve the initial koopman_phi "tau" times
        # so that we could IMPOSE a LINEAR SEQUENCE STRUCTURE IN THE LATENT SPACE

        self.future_koopman_phi_evolved_list = []

        for tau in xrange(1, self.T):
            self.future_koopman_phi_evolved_list.append(tf.matmul(self.koopmanPhi, matrix_exponential(tau * self.dt * self.koopmanOp_learn)))

        # CONCLUSION: now we have koopman phi in evolved and directly mapped from future_eta space, so we have
        # linear error ready
        # then convert each EVOLVED koopman_phi to physical space: both eta and then X
        self.future_Xrec_list = []
        self.future_etaRec_list = []
        for tau in xrange(1, self.T):
            self.future_etaRec_list.append(self.decoding_neural_net_with_pod(phi=self.future_koopman_phi_evolved_list[tau - 1],
                                                                             SVD_DMD_FLAG=self.model_params['SVD_DMD']))
            # finally note that the decoded stuff is still eta... we need to scale back to physical space
            self.future_Xrec_list.append(tf.add(self.mu_X, tf.matmul(self.future_etaRec_list[tau - 1], self.Lambda_X)))

        # CONCLUSION: now we have rec for X and eta ready... so we have reconstruction error ready...

        # note: this is only one step into future, it would be useless if in eval mode, I use single into Koopman, single time out Koopman
        self.future_Xrec_list[0] = tf.identity(self.future_Xrec_list[0], name='Xfuture')

        # build loss

        # WEIGHT ON LOSS
        normConst = tf.constant(1.0, dtype=TF_FLOAT) / (tf.constant(1.0, dtype=TF_FLOAT) + self.model_params['c_los_lin'] + self.model_params['c_los_reg'])

        # 1. RECONSTRUCTION LOSS
        # -- from an autoencoder perspective, we need Xrec and X as close as possible
        # -- from a koopman perspective, we need Xrec_future and X_future to be as close as possible

        # "rec_loss": reconstruction error in physical space
        self.rec_loss = tf.reduce_mean(tf.squared_difference(self.X, self.Xrec))

        for tau in xrange(1, self.T):
            self.rec_loss += tf.reduce_mean(tf.squared_difference(self.future_X_list[tau - 1], self.future_Xrec_list[tau - 1]))

        # "norm_rec_loss": reconstruction error in normalized space (I guess I didn't use it actually...)
        self.norm_rec_loss = tf.reduce_mean(tf.squared_difference(self.eta, self.etaRec))
        for tau in xrange(1, self.T):
            self.norm_rec_loss += tf.reduce_mean(tf.squared_difference(self.future_etaRec_list[tau - 1], self.future_eta_list[tau - 1]))

        # 2. LINEAR LOSS
        # -- from a Koopman perspective, we need the transformed latent state being as close as a linear sequence as possible
        self.lin_loss = tf.constant(0.0, dtype=TF_FLOAT)
        for tau in xrange(1, self.T):
            self.lin_loss += tf.reduce_mean(tf.squared_difference(self.future_koopman_phi_list[tau - 1], self.future_koopman_phi_evolved_list[tau - 1]))

        # REGULARIZATION LOSS

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

        ## SUMMING UP LOSS
        # First, the following `loss_up` will only be used, if you are doing the standard (deterministic) LRAN.
        # Important note
        # - it is controversial that whether to use normalized_rec_loss or rec_loss
        #   simply because the later might be more reasonable if the raw data is POD coefficients.
        #   however, un-normalized data might lead to imbalance between rec and linear lsos, reg_loss,
        #   there is might be additional need to tweak the `c_los_lin` and `c_los_reg` to balance
        #   it. One might want to plot linear loss and rec_loss separately.
        ## todo: check if linear loss and rec_loss has been plotted separately in standard LRAN

        self.loss_op = self.norm_rec_loss + self.model_params['c_los_lin'] * self.lin_loss \
                       + self.model_params['c_los_reg'] * self.reg_parameter_loss
        self.loss_op = self.rec_loss + self.model_params['c_los_lin'] * self.lin_loss \
                       + self.model_params['c_los_reg'] * self.reg_parameter_loss

        # final cost function...
        self.loss_op = normConst * self.loss_op

        ## QUICK SUMMARY
        # up until now, we have:
        # - return the mean-eta-rec loss and mean-linear-eta loss
        # - return prior for the hpp of W,b,K
        # - return the prior for W,b,K

        ## PREPARE RESIDUAL FOR BAYESIAN INFERENCE
        # VECTORIZE FOR THE LOSSES FOR EDWARD TO READ!

        residual_vector_lin_loss = [self.future_koopman_phi_list[tau - 1] - self.future_koopman_phi_evolved_list[tau - 1]
                                    for tau in xrange(1, self.T)]

        residual_vector_rec_loss = [self.future_etaRec_list[tau - 1] - self.future_eta_list[tau - 1]
                                    for tau in xrange(1, self.T)]
        residual_vector_rec_loss.append(self.etaRec - self.eta)

        residual_vector_lin_loss = tf.concat(residual_vector_lin_loss, axis=1)
        residual_vector_rec_loss = tf.concat(residual_vector_rec_loss, axis=1)

        residual_vector_lin_loss = tf.identity(residual_vector_lin_loss, name='residual_lin_loss')
        residual_vector_rec_loss = tf.identity(residual_vector_rec_loss, name='residual_rec_loss')

        return residual_vector_rec_loss, residual_vector_lin_loss

    def train(self):
        """mini-batch training the network

        """


        # randomize data before training

        random_index = self.shuffle_data_index(X=self.Xtrain)
        self.Xtrain = self.Xtrain[random_index]
        self.XfutureTrain = self.XfutureTrain[random_index]

        N_EPOCHS = self.model_params['numberEpoch']
        BATCH_SIZE = self.model_params['miniBatch']

        train_count = self.Xtrain.shape[0]

        # recording loss

        self.loss_dict = {}
        self.loss_dict['total_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['linear_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['recon_MSE'] = {'train': [], 'valid': []}
        self.loss_dict['reg_MSE'] = {'train': [], 'valid': []}

        for i in xrange(1, N_EPOCHS + 1):

            # mini-batch training

            for start, end in zip(xrange(0, train_count, BATCH_SIZE),
                                  xrange(BATCH_SIZE, train_count + 1, BATCH_SIZE)):

                # training on a subsets of training set
                # construct feed dictionary for LRAN: batch training

                feed_dict = {self.X: self.Xtrain[start:end, :]}
                for ii in xrange(len(self.future_X_list)):
                    feed_dict[self.future_X_list[ii]] = self.XfutureTrain[start:end, ii, :]

                # run training process

                _ = self.sess.run([self.train_op], feed_dict=feed_dict)

            # construct feed dictionary for LRAN: whole training

            feed_dict_whole_train = {self.X: self.Xtrain[:, :]}
            for ii in xrange(len(self.future_X_list)):
                feed_dict_whole_train[self.future_X_list[ii]] = self.XfutureTrain[:, ii, :]

            total_loss_train, linear_loss_train, recon_loss_train, reg_loss_train = self.sess.run(
                [self.loss_op, self.lin_loss, self.rec_loss, self.reg_parameter_loss],
                feed_dict=feed_dict_whole_train
            )

            # construct feed dictionary for LRAN: whole validation

            feed_dict_whole_valid = {self.X: self.Xvalid[:, :]}
            for ii in xrange(len(self.future_X_list)):
                feed_dict_whole_valid[self.future_X_list[ii]] = self.XfutureValid[:, ii, :]

            total_loss_valid, linear_loss_valid, recon_loss_valid, reg_loss_valid = self.sess.run(
                [self.loss_op, self.lin_loss, self.rec_loss, self.reg_parameter_loss],
                feed_dict=feed_dict_whole_valid
            )

            self.loss_dict['total_MSE']['train'].append(total_loss_train)
            self.loss_dict['total_MSE']['valid'].append(total_loss_valid)
            self.loss_dict['linear_MSE']['train'].append(linear_loss_train)
            self.loss_dict['linear_MSE']['valid'].append(linear_loss_valid)
            self.loss_dict['recon_MSE']['train'].append(recon_loss_train)
            self.loss_dict['recon_MSE']['valid'].append(recon_loss_valid)
            self.loss_dict['reg_MSE']['train'].append(reg_loss_train)
            self.loss_dict['reg_MSE']['valid'].append(reg_loss_valid)

            ## print the training error... and eigenvalues....

            if i % 100 == 0:
                train_loss, K = self.sess.run(
                    [self.loss_op, self.koopmanOp_learn],
                    feed_dict=feed_dict_whole_train
                )

                valid_loss = self.sess.run(
                    [self.loss_op],
                    feed_dict=feed_dict_whole_valid
                )

                print '=============='
                print ' epoch = ', i
                print 'train_loss = ', train_loss
                print 'valid_loss = ', valid_loss
                print 'koopman operator = \n', K
                print 'koopman eigenvalue = \n', LA.eig(K)[0]
                print ''

    def save_model(self):
        """Saving ``LRAN`` models with model structures, weights, biases, """

        print '===== '
        print 'show the configuration'
        pprint.pprint(self.model_params, width=1)
        print '===== '

        ## save model
        with self.graph.as_default():
            model_dir = self.dir + '/model_saved'
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            ## for D-LRAN, this is the correct way to save the model
            # save model using Tensorflow simple save
            tf.saved_model.simple_save(self.sess, model_dir,
                                       inputs={"X0": self.X},
                                       outputs={"Xrec": self.Xrec,
                                                "Xfuture": self.future_Xrec_list[0]})

            ## for B-LRAN, unfortunately saving model is not supported in edward,
            ## so I will save all the DISTRIBUTION of weight/bias/K.
            ## then create the model from scratch again....

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
                     nonlinear_rec= self.model_params['typeRecon'] == 'nonlinear',
                     nsd = self.model_params['nsd'])


if __name__ == '__main__':
    pass
