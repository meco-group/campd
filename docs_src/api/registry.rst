Registry
========

The ``campd`` framework utilizes a centralized registry and specification system to dynamically load classes and configure experiments, models, and training loops from configuration files.

System Overview
---------------

The core registry mechanism allows components to be registered dynamically. Configuration files (like YAML) leverage the :class:`~campd.utils.registry.Spec` class to instantiate these registered components or import them via their dotted path.

.. automodule:: campd.utils.registry
   :members:
   :undoc-members:
   :show-inheritance:

Available Component Registries
------------------------------

The following subsystems provide their own registries to extend CAMPD functionality.

Architectures
~~~~~~~~~~~~~

* :data:`campd.architectures.registry.MODULES`
* :data:`campd.architectures.registry.REVERSE_NETS`
* :data:`campd.architectures.registry.CONTEXT_NETS`

Data Normalization
~~~~~~~~~~~~~~~~~~

* :data:`campd.data.normalization.registry.NORMALIZATION_REGISTRY`

Experiments
~~~~~~~~~~~

* :data:`campd.experiments.registry.EXPERIMENTS`
* :data:`campd.experiments.validators.VALIDATORS`

Training Modules
~~~~~~~~~~~~~~~~

* :data:`campd.training.registry.LOSSES`
* :data:`campd.training.registry.CALLBACKS`
* :data:`campd.training.registry.SUMMARIES`
* :data:`campd.training.registry.OBJECTIVES`
