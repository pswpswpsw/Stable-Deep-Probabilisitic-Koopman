Simple example
==============


2D simple polynomial system
---------------------------

To illustrate the power of deep learning in this problem, consider the following simple example from Lusch 2017 paper.

Note that this example is also the only system known to us has the exact non-trival Koopman operator being explictly clear.






Examples of a 2D wake flow past cylinder at :math:`Re=100`
-------------------------------------------------------------

Given first **50** POD coefficients, with signal-to-noise ratio as
0, 10, 20, 30 % on the POD coefficients, we learn the Koopman operators, and eigenfunctions.

.. list-table::

   * - .. figure:: _static/eigenvalues_all_noise.png
          :width: 100 %
          :alt: Distribution of Koopman eigenvalues
          :align: center

          Figure 1. Distribution of Koopman eigenvalues

     - .. figure:: _static/pred_mean_t1000.png
          :width: 80 %
          :alt: Mean of predicted velocity mag. field at time = 10000 sec
          :align: center

          Figure 2. Mean of predicted velocity mag. field at :math:`t=1000`

     - .. figure:: _static/pred_std_t1000.png
          :width: 70 %
          :alt: Stardard deviation of predicted velocity mag. field at time = 1000 sec
          :align: center

          Figure 3. Stardard deviation of predicted velocity mag. field at :math:`t=1000`

