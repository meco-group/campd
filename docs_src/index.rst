CAMPD — API Documentation
=========================

**Context-Aware Motion Planning Diffusion**

CAMPD is a framework for training context-aware diffusion models on trajectory data
for robot motion planning. It leverages classifier-free guidance and attention mechanisms
to condition a U-Net diffusion model on structured, sensor-agnostic contextual information.

.. note::

   For the project overview, paper, and demo videos, visit the
   `project page <https://meco-group.github.io/campd/>`_.

Getting Started
---------------

Install the package:

.. code-block:: bash

   pip install .

For development with documentation tools:

.. code-block:: bash

   pip install ".[docs]"

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/core_concepts
   guide/launching
   guide/yaml_config
   guide/extending
   guide/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Packages

   api/models
   api/architectures
   api/data
   api/training
   api/experiments
   api/utils
   api/registry
