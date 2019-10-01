pyronear.datasets
=================

All datasets are subclasses of :class:`torchvision.datasets.vision.VisionDataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.

The following datasets are available:

.. contents:: Datasets
    :local:

.. currentmodule:: pyronear.datasets


OpenFire
~~~~~~~~

.. autoclass:: OpenFire
