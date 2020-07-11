#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare POD coefficients from velocity field of VTK files."""

import os
import sys
try:
    try:
        import vtkInterface as vtki
    except:
        import pyvista as vtki
except:
    print('no vtki installed')
import numpy as np

from scipy.linalg import svd
from matplotlib import pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, "../../../")


class PreparePODdata(object):
    """Performing dimension reduction via POD on multi-dimensional time series data.

    We read the VTK files using `pyvista`_ as ``numpy.ndarray``, then compute singular value
    decomposition (`SVD`_) :math:`X=U\Sigma V'`. Then given that ``X.shape = (n_time, n_dof)``,
    we reduce the ``n_dof`` simply by considering the POD coefficients:
    :math:`XV_r = U\Sigma V'V_r = U_r\Sigma_r`.

    Then we do the following three steps:

    1. We compute full SVD and save it.

    2. We load the full SVD and truncate a few then save it.

    3. We split the data into training, testing and neglect a few unphysical part in the beginning
        then save it.

    Finally, temporal evolution of components will be drawn and saved as ``png``.

    .. _VTK:
        https://vtk.org/

    .. _pyvista:
        https://github.com/pyvista

    .. _SVD:
        https://en.wikipedia.org/wiki/Singular_value_decomposition

    Args:
        vtk_folder (:obj:`str`): folder containing the ``vtk`` files.
        case_name (:obj:`str`): folder corresponding to the simulation case.

    Attributes:
        vtk_folder (:obj:`str`): folder containing the ``vtk`` files.
        case_name (:obj:`str`): folder corresponding to the simulation case.
        UV_concate_cell_array (:obj:`numpy.ndarray`): data containing vectorization of the velocity
            field.
        whole_pod_file_name (:obj:`str`): file name for the whole POD coefficient ``npz`` file.
        u (:obj:`numpy.ndarray`): :math:`U` in SVD.
        vh (:obj:`numpy.ndarray`): :math:`V'` in SVD.
        s (:obj:`numpy.ndarray`): 1D :math:`S` in SVD.

    """

    def __init__(self, vtk_folder, case_name):
        self.vtk_folder = vtk_folder
        self.case_name = case_name
        self.UV_concate_cell_array = None
        self.whole_pod_file_name = None
        self.u = None
        self.vh = None
        self.s = None

    def read_cell_data(self, fn_start):
        """Read cell data from ``vtk``

        Args:
            fn_start (:obj:`int`): the index of starting letter in
                ``ls -lrth`` corresponding to vtk filename
        """

        # We list all the files in ``vtk_folder``, then we extract the
        # vtk filename without the affix and append with list.
        # Then we sort it from small to large, since we need the vtk data
        # to be sequential in time. Finally we put the vectorized data as
        # `UV_concate_cell_array`.

        vtk_file_list = []
        file_list = os.listdir(self.vtk_folder)
        for names in file_list:
            if names.endswith(".vtk"):
                vtk_file_list.append(names[fn_start:-4])

        vtk_file_array = np.array(vtk_file_list, dtype=int)
        vtk_file_array.sort()

        uv_concate_cell_list = []
        for vtk_file in vtk_file_array:
            print 'current working on ', vtk_file

            # Note: grid is the central object in VTK where every field is added on to grid
            grid = vtki.UnstructuredGrid(self.vtk_folder + self.case_name + '_' + str(vtk_file) + \
                                         '.vtk')

            vel_cell = grid.cell_arrays['U']  # get the vector velocity

            u_cell = vel_cell[:, 0]  # get Ux
            v_cell = vel_cell[:, 1]  # get Uy

            uv_concate_cell = np.hstack((u_cell, v_cell))  # concate U,V together

            uv_concate_cell_list.append(uv_concate_cell)  # we append the snapshots of stacked state
                                                          # vector

        self.UV_concate_cell_array = np.array(uv_concate_cell_list)  # we transform it into
                                                                     # `numpy.ndarray`

    def run_plot_and_save_pod(self):
        """Perform SVD on snapshots data, i.e., :attr:`UV_concate_cell_array`, then
        save it and further plot the singular value decay.

        Compute svd using :meth:`scipy.linalg.svd`

        """
        # noinspection PyTupleAssignmentBalance
        self.u, self.s, self.vh = svd(self.UV_concate_cell_array, full_matrices=False)

        # Save pod as ``npz`` file
        np.savez(self.case_name + '_whole_POD.npz', u=self.u, s=self.s, vh=self.vh)

        # Save the singular value decay plot
        plt.figure()
        plt.loglog(self.s)
        plt.title('SVD decay of case: ' + self.case_name)
        plt.ylabel('singular values')
        plt.xlabel('number of singular values')
        plt.savefig('sv_decay.png')
        plt.close()

    def load_pod(self):
        """Load the full POD coefficient data from :attr:`whole_pod_file_name` file."""

        self.whole_pod_file_name = self.case_name + '_whole_POD.npz'
        data = np.load(self.whole_pod_file_name)
        self.u = data['u']
        self.s = data['s']
        self.vh = data['vh']

    def save_and_plot_truncated_pod_data(self, rank, dt):
        """Save the truncated POD data.

        Args:
            rank (:obj:`int`): number of ranks to keep.
            dt (:obj:`float`): time interval between snapshots.
        """

        self.load_pod()  # load full POD data

        # We truncate the full POD data by `rank`.
        # Note:: first dimension is # timesnap, second dimension is # state space.

        u_r = self.u[:, :rank]
        vh_r = self.vh[:rank, :]
        s_r = self.s[:rank]

        # We save truncated POD data as with ``_rank_`` in the filename

        np.savez(self.case_name + '_rank_' + str(rank) + '_POD.npz', u_r=u_r, vh_r=vh_r, s_r=s_r)

        # We plot POD coefficients with repsect to time and save it

        u_r = self.u[:, :rank]  # first dimension is # timesnap, second dimension is # state space.
        s_r = self.s[:rank]

        a_r = np.matmul(u_r, np.diag(s_r))  # get the temporal modes

        num_data = u_r.shape[0]  # total number of data in the POD results!

        fig, ax = plt.subplots(rank, figsize=(128, 100))
        for ind in range(rank):
            ax[ind].plot(np.arange(num_data) * dt, a_r[:, ind], color='k')
            ax[ind].set_ylabel('unscaled POD coefficients', color='k')
            ax[ind].set_xlabel('time')
        plt.title('full length unscaled POD coefficients vs time: ' + self.case_name + ' rank = ' + \
                  str(rank))
        plt.savefig(self.case_name + '_full_length_POD_unscaled_coeffcients_vs_time.png')
        plt.close()

        # We save POD coefficients into npz file.
        np.savez(self.case_name + '_rank_' + str(rank) + '_RAW_POD_WHOLE.npz', a_r=a_r)

    def generate_training_testing_data(self, rank, dt, init_perc_train, end_perc_train,
                                       end_perc_whole, subsample_factor=1):
        """Generate train/test data and plot the temporal evolution then save it into ``npz`` file.

        Args:
            rank (:obj:`int`): number of POD modes to keep.
            dt (:obj:`float`): time interval between snapshots.
            init_perc_train (:obj:`float`): the percentage of the begining of training data.
            end_perc_train (:obj:`float`): the percentage of the ending of training data.
            end_perc_whole (:obj:`float`): the percentage of the ending of whole data.
            subsample_factor (:obj:`int`): ratio of subsampling. 2 means sampling every other
                snapshot. Default value is 1.

        """
        # We read truncated POD coefficients.
        data_pod_coef = np.load(self.case_name + '_rank_' + str(rank) + '_RAW_POD_WHOLE.npz')

        a_r = data_pod_coef['a_r']

        num_data = a_r.shape[0]

        index_init_train = int(init_perc_train * num_data)
        num_data_whole = int(num_data * end_perc_whole)

        # Print out information about time interval in the final data.
        if subsample_factor > 1:
            print 'subsampled: dt = ', float(dt * subsample_factor)
        else:
            print 'not subsampled: dt = ', dt

        # We subsampling the data depends on the `subsampling_factor`.
        whole_index = np.arange(index_init_train, num_data_whole, subsample_factor)

        # We compute the ending index for training in the whole data so we can cut off too long
        # signals.
        index_end_train = int(end_perc_train * whole_index.shape[0])

        train_index = whole_index[0:index_end_train]
        test_index = whole_index[index_end_train:]

        a_r_train = a_r[train_index]
        a_r_test = a_r[test_index]
        # noinspection PyTypeChecker
        a_r_whole = a_r[whole_index]

        # We save the training data, testing data, whole data with rank truncations as npz.
        np.savez(self.case_name + '_rank_' + str(rank) + '_POD_training.npz', Xtrain=a_r_whole)
        np.savez(self.case_name + '_rank_' + str(rank) + '_POD_testing.npz', Xtrain=a_r_test)
        np.savez(self.case_name + '_rank_' + str(rank) + '_POD_whole.npz', Xtrain=a_r_train)

        print 'number of training data = ', a_r_train.shape[0]
        print 'number of testing data = ', a_r_test.shape[0]
        print 'number of total data = ', a_r_whole.shape[0]

        # We save the plot for truncated temporal modes, i.e., POD coefficients.
        fig, ax = plt.subplots(rank, figsize=(128, 100))
        for ind in range(rank):
            ax[ind].plot(whole_index * dt, a_r_whole[:, ind], color='k')
            ax[ind].set_ylabel('unscaled POD coefficients', color='k')
            ax[ind].set_xlabel('time')
        plt.title('whole length unscaled POD coefficients vs time: ' + self.case_name + ' rank = ' \
                  + str(rank))
        plt.savefig(self.case_name + '_whole_length_POD_unscaled_coeffcients_vs_time.png')
        plt.close()


if __name__ == '__main__':
	pass