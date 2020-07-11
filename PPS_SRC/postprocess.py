#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modules for postprocessing.

Note that I used my own ``plt.style`` as ``siads`` style.
"""

import sys
sys.dont_write_bytecode = True
sys.path.insert(0,'../')
import numpy as np
import os

# import vtki
try:
    try:
        import vtkInterface as vtki
    except:
        import pyvista as vtki
except:
    print('no vtki installed')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PREP_DATA_SRC.source_code.lib.utilities import mkdir

plt.style.use('siads')

class ClassKoopmanPPS(object):
    """
    Base Class postprocessing for Koopman Deep Learning.

    Args:
        pps_dir (:obj:`str`): path to the postprocessing folder. Plots are saved here.

        eval_dir (:obj:`str`): path to the eval data folder.

        model_dir (:obj:`str`): path to the original model folder.

        params (:obj:`dict`): additional params for plotting.

        mode (:obj:`str`): strings for determining the mode of postprocessing. It can be ``nonBayes`` or
            ``Standard``.

    Attributes:

        pps_dir (:obj:`str`): path to the postprocessing folder. Plots are saved here.

        eval_dir (:obj:`str`): path to the eval data folder.

        model_dir (:obj:`str`): path to the original model folder.

        params (:obj:`dict`): additional params for plotting.

        mode (:obj:`str`): strings for determining the mode of postprocessing. It can be ``nonBayes`` or
            ``Standard``.

    """

    def __init__(self, pps_dir, eval_dir, model_dir, params, mode='nonBayes'):

        self.pps_dir = pps_dir
        self.eval_dir = eval_dir
        self.model_dir = model_dir
        mkdir(self.pps_dir)
        self.params = params
        self.mode = mode

    def pps_eigenfunction(self):
        raise NotImplementedError("need to implement eigfunction plotting")

    def pps_eigenvalues(self):
        raise NotImplementedError("Postprocessing for eigenvalues need to be implemented")

    def pps_eigenmodes_eval(self):
        raise NotImplementedError("Postprocessing for eigenmodes evaluation need to be implemented")

    def pps_paraview_cylinder(self, cylinder_folder="./50d_cylinder_flow_pod_case",
        pod_path="/media/shaowu/shaowu_main_hard/shaowu/OpenFOAM/shaowu-6/run/vortex_shedding/c6_re100_whole_POD.npz"):
        """postprocessing for cylinder case with paraview.

        This is a very problem specific method. It will use the POD weights and it will use eval trajectory
        to get the full field projected and saved into vtk files. It is a good example for extending to other cases.

        Args:
            cylinder_folder (:obj:`str`): path to the full POD data.

        """

        # 1. read POD data from POD path
        pod_data = np.load(pod_path)
        vh = pod_data['vh']
        # s = pod_data['s']
        # u = pod_data['u']

        # 2. read eval file
        eval_file_name = 'save_trj_comparison_whole.npz'
        eval_file_data = np.load(self.eval_dir + '/' + self.params['model_name'] + '/' + eval_file_name)
        true_trajectory = eval_file_data['ttrj']
        pred_trajectory = eval_file_data['ptrj']

        # compute the mean and std
        print('')
        print('pred trajectory shape = ')
        print(pred_trajectory.shape)
        print('')

        pred_trajectory_mean = np.mean(pred_trajectory, axis=0)
        pred_trajectory_std  = np.std( pred_trajectory, axis=0)
        print('pred_trajectory_std shape = ', pred_trajectory_std.shape)

        # 3. generate VTK files
        self.vtk_folder = self.pps_dir+  '/VTK'

        # make directory for vtk
        mkdir(self.vtk_folder)

        # 3.1 generate true flowfield VTK
        true_data_UV_array = np.matmul(true_trajectory, vh[:true_trajectory.shape[1],:])
        true_data_U = true_data_UV_array[:,:int(true_data_UV_array.shape[1]/2) ]
        true_data_V = true_data_UV_array[:, int(true_data_UV_array.shape[1]/2):]
        true_data_W = np.zeros(true_data_U.shape)

        # get 3d true velo
        true_data_vel = np.stack((true_data_U, true_data_V, true_data_W), axis=2)

        # 3.2 generate pred flowfield VTK -- mean
        pred_data_UV_array = np.matmul(pred_trajectory_mean, vh[:pred_trajectory_mean.shape[1],:])
        pred_data_U = pred_data_UV_array[:, :int(pred_data_UV_array.shape[1]/2) ]
        pred_data_V = pred_data_UV_array[:,  int(pred_data_UV_array.shape[1]/2):]
        pred_data_W = np.zeros(pred_data_U.shape)

        # get 3d pred velo
        pred_data_vel_mean = np.stack((pred_data_U, pred_data_V, pred_data_W), axis=2)

        # 3.2 -- std
        pred_data_UV_array_std = np.matmul(pred_trajectory_std, vh[:pred_trajectory_std.shape[1],:])
        pred_data_U_std = pred_data_UV_array_std[:,:int(pred_data_UV_array_std.shape[1]/2) ]
        pred_data_V_std = pred_data_UV_array_std[:, int(pred_data_UV_array_std.shape[1]/2):]
        pred_data_W_std = np.zeros(pred_data_U_std.shape)

        # get 3d pred velo -- std
        pred_data_vel_std = np.stack((pred_data_U_std, pred_data_V_std, pred_data_W_std), axis=2)

        # 3.5 read blank VTK files for cylinder to write new things on: loop over time
        for time_step in range(true_data_U.shape[0]):

            true_data = vtki.UnstructuredGrid(cylinder_folder + '/base.vtk')
            pred_data = vtki.UnstructuredGrid(cylinder_folder + '/base.vtk')

            # 3.2 add fields in
            true_data.AddCellScalars(true_data_vel[time_step,:,:], 'U')

            pred_data.AddCellScalars(pred_data_vel_mean[time_step,:,:], 'U_mean')
            pred_data.AddCellScalars( pred_data_vel_std[time_step,:,:], 'U_std' )

            # save
            true_data.Write(self.vtk_folder + '/true_flowfield_' + str(time_step) + '.vtk')
            pred_data.Write(self.vtk_folder + '/pred_flowfield_' + str(time_step) + '.vtk')

            del(true_data)
            del(pred_data)

    def pps_component_trj(self, additional_names=None, alpha=0.5, ylim=(-70,70), fig_size=(8,8), lw=1):
        """ postprocessing each component trajectory

        Args:

            additional_names (:obj:`str`): string for figures to be saved in a more specific name.

            alpha (:obj:`float`): level of transparent

            ylim: (:obj:`numpy.ndarray`): ylim of figure

            fig_size: (:obj:`tuple`): figure size

            lw: (:obj:`float`): line width.

        """

        # LOAD SaveCompareWithTruth
        if additional_names == None:
            file_name = 'save_trj_comparison.npz'
            output_name = '_'
        else:
            file_name = 'save_trj_comparison_' + additional_names +  '.npz'
            output_name = additional_names
        fig_data = np.load(self.eval_dir + '/' + self.params['model_name'] + '/' + file_name)
        true_trajectory = fig_data['ttrj']
        pred_trajectory = fig_data['ptrj']
        true_tsnap = fig_data['tt'] - np.min(fig_data['tt'])

        # # plot for each component
        if self.mode == 'nonBayes':

            num_components = true_trajectory.shape[1]
            for i_comp in range(num_components):
                if additional_names == None:
                    plt.figure(figsize=fig_size)
                else:
                    plt.figure(figsize=fig_size)
                plt.plot(true_tsnap, true_trajectory[:, i_comp], 'k-', label='true')
                plt.plot(true_tsnap, pred_trajectory[:, i_comp], 'r--',label='pred')

                plt.xlabel('time')
                plt.ylabel(r'$x_{' + str(i_comp + 1) + '}$',fontsize = 32)
                lgd = plt.legend(bbox_to_anchor=(1, 0.5))
                plt.savefig(self.pps_dir + \
                            '/component_' + str(i_comp + 1) + '_' + output_name + '.png',  bbox_extra_artists=(lgd,),
                            bbox_inches='tight')
                plt.close()

            ## plot for total components
            if additional_names == None:
                plt.figure(figsize=fig_size)
            else:
                plt.figure(figsize=fig_size)
            plt.xlabel('time')
            plt.ylabel(r'$x$')
            for i_comp in range(num_components):
                plt.plot(true_tsnap, pred_trajectory[:, i_comp], 'r--',label='pred: ' + str(i_comp+1) )
                plt.plot(true_tsnap, true_trajectory[:, i_comp], 'k-', label='true: ' + str(i_comp+1) )
            # lgd = plt.legend(bbox_to_anchor=(1, 0.5))
            plt.savefig(self.pps_dir + \
                            '/total_component_' + output_name + '.png', bbox_inches='tight')
            plt.close()

        elif self.mode == 'Bayes':

            num_components = true_trajectory.shape[1]

            num_samples = pred_trajectory.shape[0]

            for i_comp in range(num_components):

                plt.figure(figsize=fig_size)
                for i_sample in range(num_samples):
                    if i_sample == 0:
                        plt.plot(true_tsnap, pred_trajectory[i_sample, :, i_comp], 'r-', linewidth=lw, label='Monte Carlo', alpha=alpha)
                    else:
                        plt.plot(true_tsnap, pred_trajectory[i_sample, :, i_comp], 'r-', linewidth=lw, alpha=alpha)
                plt.plot(true_tsnap, true_trajectory[:, i_comp], 'k-', label='true')

                # plt.plot(true_tsnap, pred_trajectory[:, i_comp], 'r--',label='pred')
                plt.xlabel('time',fontsize = 32)
                plt.ylabel(r'$x_{' + str(i_comp + 1) + '}$',fontsize = 32)
                # lgd = plt.legend(bbox_to_anchor=(1, 0.5))
                plt.savefig(self.pps_dir + '/component_' + str(i_comp + 1) + '_' + output_name + '.png', #  bbox_extra_artists=(lgd,),
                            bbox_inches='tight')
                plt.close()

            ## plot for total components
            if additional_names == None:
                plt.figure(figsize=fig_size)
            else:
                plt.figure(figsize=fig_size)
            plt.xlabel('time',fontsize = 32)
            plt.ylabel(r'$x$',fontsize = 32)
            for i_comp in range(num_components):
                if i_comp != 0:
                    for i_sample in range(num_samples):

                        if i_sample == 0:

                            # skip for modes
                            plt.plot(true_tsnap, pred_trajectory[i_sample, :, i_comp], 'r-', linewidth=lw,  label='Monte Carlo' + str(i_comp+1), alpha=alpha )
                        else:
                            plt.plot(true_tsnap, pred_trajectory[i_sample, :, i_comp], 'r-', linewidth=lw, alpha=alpha)

                        plt.plot(true_tsnap, true_trajectory[:, i_comp], 'k-', label='true: ' + str(i_comp + 1))

                    plt.ylim(ylim)

                # plt.plot(true_tsnap, pred_trajectory[:, i_comp], 'r--',label='pred: ' + str(i_comp+1) )
            # lgd = plt.legend(bbox_to_anchor=(1, 0.5))
            plt.savefig(self.pps_dir + '/total_component_' + output_name + '.png', bbox_inches='tight')
            plt.close()

    def pps_2d_data_dist(self, data_path):
        """postprocessing for the training data for 2D system.

        Args:

            data_path (:obj:`str`): path to the data files.

        """
        # only obtain the phase space locations
        data2D = np.load(data_path)['Xtrain']

        plt.figure()
        plt.scatter(data2D[:,0], data2D[:,1], s=0.1 ,c='k')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(
            self.pps_dir + '/trainPhaseDist.png',
            bbox_inches='tight'
        )
        plt.close()

    def get_cmap(self, n, name='rainbow'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)



class ClassDLPPS(ClassKoopmanPPS):
    """
    Class for Postprocessing of Deep Learning Koopman.

    Previously I have a class that is only focusing on the EDMD/KDMD.

    """
    def pps_eigenvalues(self, mode_force=None):
        """plot eigenvalues

        Args:
            mode_force (:obj:`bool`): we can force to use the eigenvalues from eval_dir or model_dir

        """

        if mode_force ==None:
            if self.mode == 'nonBayes':

                ## LOAD koopman_eigenvalue_deeplearn
                fig_data = np.load(self.model_dir + '/../koopman_eigenvalue_deeplearn.npz')
                D_real = fig_data['eig_real']
                D_imag = fig_data['eig_imag']

            elif self.mode == 'Bayes':

                fig_data = np.load(self.eval_dir + '/' + self.params['model_name'] + '/koopman_eigenfunctions.npz')

                D_real = fig_data['D_real'].reshape(-1,1)
                D_imag = fig_data['D_imag'].reshape(-1,1)
        else:
            ## LOAD koopman_eigenvalue_deeplearn
            fig_data = np.load(self.model_dir + '/../koopman_eigenvalue_deeplearn.npz')
            D_real = fig_data['eig_real']
            D_imag = fig_data['eig_imag']

        ## plot eigenvalues
        plt.figure()
        # draw eigenvalue
        if self.mode == 'nonBayes':
            plt.scatter(D_real, D_imag, c='b')
        elif self.mode == 'Bayes':
            plt.scatter(D_real.flatten(), D_imag.flatten(), c='b', s=1, alpha=0.1)
        # draw circle
        ## no need, since it is a continuous Koopman eigenvalue..
        # theta = np.linspace(0, 2*np.pi ,200)
        # plt.plot(np.cos(theta), np.sin(theta), 'k-')

        plt.xlabel(r'Real($\lambda$)')
        plt.ylabel(r'Imag($\lambda$)')
        # lgd = plt.legend(['train', 'validation'], bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/koopmanEigVal.png', bbox_inches='tight')

        print('range of K: real part', np.min(D_real), np.max(D_real))
        print('std of K: real part', np.std(D_real))
        print('range of K: imag part', np.min(D_imag), np.max(D_imag))
        print('std of K: imag part', np.std(D_imag))

        plt.close()

    def pps_eigenfunction(self):
        """
        plot eigenfunctions over 2D system
        """
        if self.mode == 'nonBayes':

            ## load
            fig_data = np.load(self.model_dir + '/../koopman_eigenfunctions.npz')

            numKoopmanModes = fig_data['numKoopmanModes']
            R = fig_data['R']
            phi_array = fig_data['phi_array']
            ndraw = fig_data['ndraw']
            D_real = fig_data['D_real']
            D_imag = fig_data['D_imag']
            x1_ = fig_data['x1_']
            x2_ = fig_data['x2_']

            # draw koopman eigenfunction
            # note that \phi as a row vector, mulitples R1, R2 will obtain phi_eigen 1 and phi_eigen 2
            for ikoopman in range(numKoopmanModes):
                R_i = R[:, ikoopman:ikoopman + 1]
                phi_eigen = np.matmul(phi_array, R_i)
                phi_eigen_mesh = phi_eigen.reshape((ndraw, ndraw))

                # draw || mag. of eigenfunction
                plt.figure()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.title(
                    r'$\lambda$ = ' + "{0:.3f}".format(D_real[ikoopman]) + ' + ' + "{0:.3f}".format(D_imag[ikoopman]) + 'i')
                plt.contourf(x1_, x2_, np.abs(phi_eigen_mesh), 100, cmap=plt.cm.get_cmap('jet'))
                plt.colorbar()
                plt.savefig(self.pps_dir + '/koopmanEigFunct_MAG_mode_' + str(ikoopman) + '.png', bbox_inches='tight')
                plt.close()

                # draw phase angle of eigenfunction
                plt.figure()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.title(
                    r'$\lambda$ = ' + "{0:.3f}".format(D_real[ikoopman]) + ' + ' + "{0:.3f}".format(D_imag[ikoopman]) + 'i')
                plt.contourf(x1_, x2_, np.angle(phi_eigen_mesh), 100, cmap=plt.cm.get_cmap('jet'))
                plt.colorbar()
                plt.savefig(self.pps_dir + '/koopmanEigFunct_ANG_mode_' + str(ikoopman) + '.png', bbox_inches='tight')
                plt.close()

        elif self.mode == 'Bayes':

            ## load
            fig_data = np.load(self.eval_dir + '/' + self.params['model_name'] + '/koopman_eigenfunctions.npz')

            numKoopmanModes = fig_data['numKoopmanModes']
            R = fig_data['R']

            num_samples = R.shape[0]

            phi_array = fig_data['phi_array']
            ndraw = fig_data['ndraw']
            D_real = fig_data['D_real']
            D_imag = fig_data['D_imag']
            x1_ = fig_data['x1_']
            x2_ = fig_data['x2_']

            # loop over all realization, to get the mean abs and angle..

            # draw koopman eigenfunction
            # note that \phi as a row vector, mulitples R1, R2 will obtain phi_eigen 1 and phi_eigen 2
            for ikoopman in range(numKoopmanModes):

                abs_phi_eigen_mean = 0
                for i_sample in range(num_samples):
                    R_i = R[i_sample, :, ikoopman:ikoopman + 1]
                    phi_eigen = np.matmul(phi_array[i_sample], R_i)
                    abs_phi_eigen_mean += np.abs(phi_eigen)

                abs_phi_eigen_mean /= num_samples

                abs_phi_eigen_std = 0
                for i_sample in range(num_samples):
                    R_i = R[i_sample, :, ikoopman:ikoopman + 1]
                    phi_eigen = np.matmul(phi_array[i_sample], R_i)
                    abs_phi_eigen_std += (np.abs(phi_eigen) - abs_phi_eigen_mean)**2

                abs_phi_eigen_std /= num_samples
                abs_phi_eigen_std = np.sqrt(abs_phi_eigen_std)

                # convert it into the grid
                abs_phi_eigen_mean_mesh = abs_phi_eigen_mean.reshape((ndraw, ndraw))
                abs_phi_eigen_std_mesh = abs_phi_eigen_std.reshape((ndraw, ndraw))

                ### start drawing contour of mean phi and normalized uncertainty level of of phi

                plt.figure(figsize=(12,8))
                plt.contourf(x1_, x2_, abs_phi_eigen_mean_mesh, 100, linestyles="solid" ,cmap='jet')
                plt.colorbar()
                plt.xlabel(r'$x_1$',fontsize = 32)
                plt.ylabel(r'$x_2$',fontsize = 32)
                plt.savefig(self.pps_dir + '/koopmanEigFunct_MAG_mean_mode_' + str(ikoopman) + '.png',
                            bbox_inches='tight')
                plt.close()


                v = np.linspace(1, 10, 9, endpoint=False)

                plt.figure(figsize=(12,8))
                plt.contourf(x1_, x2_, abs_phi_eigen_std_mesh/np.min(abs_phi_eigen_std_mesh), v, linestyles="solid",cmap='jet')
                plt.colorbar(ticks=v)
                plt.xlabel(r'$x_1$',fontsize = 32)
                plt.ylabel(r'$x_2$',fontsize = 32)
                plt.savefig(self.pps_dir + '/koopmanEigFunct_MAG_normalized_uncertainty_mode_' + str(ikoopman) + '.png',
                            bbox_inches='tight')
                plt.close()

                ### start drawing abs of phi

                fig = plt.figure(figsize=(12,8))
                ax = fig.add_subplot(111, projection="3d")

                # surface plot with mean and -+2 sigma
                Z_mean = abs_phi_eigen_mean_mesh
                Z_plus_3s = abs_phi_eigen_mean_mesh + 3 * abs_phi_eigen_std_mesh
                Z_mins_3s = abs_phi_eigen_mean_mesh - 3 * abs_phi_eigen_std_mesh

                # ax.plot_surface(x1_, x2_, np.log(abs_phi_eigen_std_mesh/np.min(abs_phi_eigen_std_mesh)), cmap="gray", alpha=1)
                surface = ax.plot_surface(x1_, x2_, Z_mean, cmap="magma")
                ax.plot_surface(x1_, x2_, Z_plus_3s, cmap="gray", alpha=0.5)
                ax.plot_surface(x1_, x2_, Z_mins_3s, cmap="gray", alpha=0.5)

                offset_max = np.max(Z_plus_3s)
                offset_min = np.min(Z_mins_3s)

                ax.contour(x1_, x2_, Z_mean, 20, linestyles="solid", offset=offset_max*1.05)

                # debug
                # ax.set_zlim(0, 10)
                # ax.set_zlim(offset_min, offset_max*1.05)

                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Now set color to white (or whatever is "invisible")
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')

                # ax.contourf(x_m, y_m, x_m)
                ax.set_xlabel(r'$x_1$', fontsize=20)
                ax.set_ylabel(r'$x_2$', fontsize=20)
                ax.zaxis.set_rotate_label(False)
                ax.set_zlabel(r'$|\phi|$', fontsize=20, labelpad=10)

                # plt.title(r'$\lambda$ = ' + "{0:.3f}".format(float(D_real[ikoopman])) + ' + ' + "{0:.3f}".format(float(D_imag[ikoopman])) + 'i')
                plt.colorbar(surface)

                plt.savefig(self.pps_dir + '/koopmanEigFunct_MAG_TOTAL_uncertainty_mode_' + str(ikoopman) + '.png',
                            bbox_inches='tight')

                # plt.show()

                plt.close()



                ### start drawing angle of phi

                phi_eigen_mean = 0
                for i_sample in range(num_samples):
                    R_i = R[i_sample, :, ikoopman:ikoopman + 1]
                    phi_eigen = np.matmul(phi_array[i_sample], R_i)
                    phi_eigen_mean += np.angle(phi_eigen)

                phi_eigen_mean /= num_samples

                phi_eigen_std = 0
                for i_sample in range(num_samples):
                    R_i = R[i_sample, :, ikoopman:ikoopman + 1]
                    phi_eigen = np.matmul(phi_array[i_sample], R_i)
                    phi_eigen_std += (np.angle(phi_eigen) - phi_eigen_mean)**2

                phi_eigen_std /= num_samples
                phi_eigen_std = np.sqrt(phi_eigen_std)

                # convert it into the grid
                phi_eigen_mean_mesh = phi_eigen_mean.reshape((ndraw, ndraw))
                phi_eigen_std_mesh = phi_eigen_std.reshape((ndraw, ndraw))


                plt.figure(figsize=(12,8))
                plt.contourf(x1_, x2_, phi_eigen_mean_mesh, 100, linestyles="solid",cmap='jet')
                plt.colorbar()
                plt.xlabel(r'$x_1$',fontsize = 32)
                plt.ylabel(r'$x_2$',fontsize = 32)
                plt.savefig(self.pps_dir + '/koopmanEigFunct_mean_ANG_mode_' + str(ikoopman) + '.png', bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(12,8))
                plt.contourf(x1_, x2_, phi_eigen_std_mesh, 100, linestyles="solid",cmap='jet')
                plt.colorbar()
                plt.xlabel(r'$x_1$',fontsize = 32)
                plt.ylabel(r'$x_2$',fontsize = 32)
                plt.savefig(self.pps_dir + '/koopmanEigFunct_std_ANG_mode_' + str(ikoopman) + '.png',
                            bbox_inches='tight')
                plt.close()


    def pps_learning_curve(self):
        """plot learning curve of the training process
        """

        # loop through all learning curves
        for prefix in ['total-MSE','linear-MSE','recon-MSE']:
            if self.mode == 'Bayes' and prefix != 'total-MSE':
                break
            ## LOAD data
            fig_data = np.load(self.model_dir + '/../' + prefix + '_learn_curve.npz')
            train_metrics_list = fig_data['train_metrics_list']
            valid_metrics_list = fig_data['valid_metrics_list']

            # plot learning curve
            plt.figure()
            plt.semilogy(range(len(train_metrics_list)), train_metrics_list, 'b-', alpha=0.5)
            plt.semilogy(range(len(train_metrics_list)), valid_metrics_list, 'r-', alpha=0.5)
            plt.xlabel('epoch')
            plt.ylabel(prefix)
            lgd = plt.legend(['train','validation'], bbox_to_anchor=(1, 0.5))
            plt.savefig(self.pps_dir + '/' + prefix + '.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()

