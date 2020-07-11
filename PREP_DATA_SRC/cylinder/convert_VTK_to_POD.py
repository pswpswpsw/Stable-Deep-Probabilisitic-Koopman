#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example: prepare POD coefficients for 2D flow past cylinder.

We collect the data in vtk format from OpenFOAM. Then we perform SVD.
Refer to `PREP_DATA_SRC.PreparePODdata`. Please put this code in a separate
folder that shares the same directory with the location where OpenFOAM
VTK data lies.

This code will generate the following:

1. sv_decay.png: decay of singular values.

2. `case_name`_whole_POD.npz

3. `case_name`_rank_%d_POD.npz

4. `case_name`_rank_%d_RAW_POD_WHOLE.npz

5. `case_name`_rank_%d_POD_training.npz: truncated training npz file.

6. `case_name`_rank_%d_POD_testing.npz: truncated testing npz file.

7. `case_name`_rank_%d_POD_whole.npz: truncated whole npz file.

8 . `case_name`_whole_length_POD_unscaled_coeffcients_vs_time.png: plot of components
    for whole POD coefficients.

"""

import sys
sys.dont_write_bytecode = True
sys.path.append("../")

from PREP_DATA_SRC.PreparePODdata import *

def main():

    # We setup the folder where vtk is stored and the case name for OpenFOAM

    vtk_folder_name = '../../c6_re100/VTK/' # location of VTK files
    case_name = "c6_re100" # case name

    # We set the starting index of the filename in ls -lrth.
    # This depends on the actual length of case_name.

    range_index = 9

    rank = 50 #: choose SVD reduced rank
    dt = 0.1

    first_time = False #: if first time, you need to save POD data first
    # first_time = True

    init_perc_train = 0.03
    end_perc_train = 0.4
    end_perc_whole= 0.6
    subsample_factor = 2

    if first_time:

        # prepare full pod

        data_converter = PreparePODdata(vtk_folder=vtk_folder_name, case_name=case_name)
        data_converter.read_cell_data(fn_start=range_index)
        data_converter.run_plot_and_save_pod()

    else:

        # truncate to get the ideal POD data

        data_converter = PreparePODdata(vtk_folder=vtk_folder_name, case_name=case_name)
        data_converter.load_pod()
        data_converter.save_and_plot_truncated_pod_data(rank=rank, dt=dt)
        data_converter.generate_training_testing_data(rank, dt, init_perc_train, end_perc_train, end_perc_whole, subsample_factor=subsample_factor)


if __name__ == '__main__':
    main()
