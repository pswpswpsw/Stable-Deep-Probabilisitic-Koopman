.. Deep learning Koopman documentation master file, created by
   sphinx-quickstart on Sat May 12 21:29:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _home_page:

*************************************************************************
SPARK: Stabilized ProbAbilistic deep leaRning Koopman
*************************************************************************

.. figure:: _static/logo.png
   :width: 50%
   :alt: SPARK
   :align: center


This code consists of **four** :ref:`modules`:

* :ref:`PREP_DATA_SRC`

* :ref:`MODEL_SRC`

* :ref:`EVAL_SRC`

* :ref:`PPS_SRC`






Contents
---------

.. toctree::
   :maxdepth: 2

   modules
   tutorial

.. _requirements:

Requirements
------------

.. _python 2.7: https://github.com/python/cpython/tree/2.7

.. _gputil: https://github.com/anderskm/gputil

.. _scipy: https://github.com/scipy/scipy

.. _pyDOE: https://pythonhosted.org/pyDOE/

.. _edward 1.3.5: https://github.com/blei-lab/edward

.. _tensorflow 1.8: https://github.com/tensorflow/tensorflow

It requires a GPU card on the system and Linux 64 bits is preferred.

* `python 2.7`_
* `gputil`_
* `scipy`_
*  pyDOE_
* `edward 1.3.5`_
* `tensorflow 1.8`_


Setup on Workstation
--------------------

* issues with `edward 1.3.5`_
   * missing ``set_shapes_for_outputs`` can be resolved in `here <https://github.com/blei-lab/edward/issues/893#issuecomment-388792874>`_
   * index out of range error with `tensorflow 1.8`_ can be resolved `in here <https://github.com/blei-lab/edward/issues/895#issuecomment-396561955>`_


Setup on `Conflux <https://arc-ts.umich.edu/conflux/>`_
---------------------------------------------------------

* **_EXTERNAL_PACKAGES/** folder contain the modified source code for `gputil`_, `pyDOE`_, `edward 1.3.5`_ and corrsesponding installed version

* simply append the ``~/.bashrc`` with, and **remember to change to your own path**

.. code-block:: shell

   ## Configuration for tensorflow 1.8.0 + Edward
   module load compilers/gcc/5.4.0
   module load atlas/3.10.3/gcc/5.4.0

   module load lapack/3.6.1/gcc-5.4.0
   module load cuda/9.2
   module load tensorflow/1.8
   module load python-anaconda2/2018

   ## environment for python packages
   export PYTHONPATH="${PYTHONPATH}:/gpfs/gpfs0/groups/duraisamy/shawnpan/2018_bayes_dl_koopman_ode/edward_conflux/lib/python2.7/site-packages/"
   export PYTHONPATH="${PYTHONPATH}:/gpfs/gpfs0/groups/duraisamy/shawnpan/2018_bayes_dl_koopman_ode/pyDOE_conflux/lib/python2.7/site-packages/"
   export PYTHONPATH="${PYTHONPATH}:/gpfs/gpfs0/groups/duraisamy/shawnpan/2018_bayes_dl_koopman_ode/gputil_conflux/lib/python2.7/site-packages/"


Methodology
---------------------

We leverage **Mean-field Variational Inference**  inferring a
**Bayesian neural network** to learn the **Koopman eigenfunctions**, **eigenvalues**,
for a continuous nonlinear dynamical systems as :math:`\dot{x} = F(x)`.

.. _publication: https://epubs.siam.org/doi/abs/10.1137/19M1267246

For detailed information, please refer to our publication_.




:Authors:
    Shaowu Pan

:Verion: 1.3rc1 of July 10, 2020



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
