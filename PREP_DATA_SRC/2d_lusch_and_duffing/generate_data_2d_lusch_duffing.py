#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example: generate state space data from known physics for 2D toy problems.

One can keep adding toy cases where by doing the following:

1. In `MODEL_SRC.lib.model`, write your own F function that governs the
    velocity of the dynamical systems.

2. Setup the ``phase_space_range`` and ``num_samples_each_dim``.

3. write another if-clause below and change ``case``.

Attributes:
    case (:obj:`str`): name for the problem case

"""

import sys
import numpy as np

from MODEL_SRC.lib.model import F_simple_2d_system
from MODEL_SRC.lib.model import F_duffing_2d_system
from MODEL_SRC.ClassGenerateDataFromPhysics import ClassGenerateXXDotFromPhysics

sys.dont_write_bytecode = True
sys.path.insert(0, "../../")

case = '2d_simple_lusch2017'
# case = '2d_duffing_otto2017'

def main():

    if case == '2d_simple_lusch2017':

        # We set up the case with a 2D toy example on `Lusch paper`_
        # Especially the phase_space_range, and num_samples_each_dim.
        # Original case was set as 40 data points on each dimension.
        #
        # .. _Lusch paper: https://www.nature.com/articles/s41467-018-07210-0/

        noise_level = 0
        phase_space_range = [[-.5, .5], [-.5, .5]]  #
        num_samples_each_dim = 100
        range_of_X = np.array(phase_space_range)

        # Then we use ``ClassGenerateXXDotFromPhysics`` to make corresponding directory
        # and generate the data files then save.

        data_generator = ClassGenerateXXDotFromPhysics(directory='../../EXAMPLES/', case_name=case, noise_level=noise_level)
        data_generator.make_case_dir()
        data_generator.samplingX_Xdot(F=F_simple_2d_system, range_of_X=range_of_X, num_samples_each_dim=num_samples_each_dim)
        data_generator.save_X_Xdot()

    if case == '2d_duffing_otto2017':

        # We set up the case with 2D toy examples on Duffing case in `Otto's paper`_
        # with range from -2 to +2 in both x and y direction. We assign 100 data
        # on each dimension so it is 10000 data in total.
        #
        # .. Otto's paper: https://arxiv.org/abs/1712.01378

        noise_level = 0
        phase_space_range = [[-2, 2], [-2, 2]]
        num_samples_each_dim = 100
        range_of_X = np.array(phase_space_range)

        # Then we setup the case with ``ClassGenerateXXDotFromPhysics`` to make directory
        # and generate data files and save.

        data_generator = ClassGenerateXXDotFromPhysics(directory='../../EXAMPLES/', case_name=case, noise_level=noise_level)
        data_generator.make_case_dir()
        data_generator.samplingX_Xdot(F=F_duffing_2d_system, range_of_X=range_of_X, num_samples_each_dim=num_samples_each_dim)
        data_generator.save_X_Xdot()


if __name__ == '__main__':
    main()
