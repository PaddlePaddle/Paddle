Use different clusters
======================

The user's cluster environment is not the same. To facilitate everyone's deployment, we provide a variety of cluster deployment methods to facilitate the submission of cluster training tasks, which will be introduced as follows:

`Kubernetes <http://kubernetes.io>`_ is a scheduling framework of Google open source container cluster, supporting a complete cluster solution for large-scale cluster production environment. The following guidelines show PaddlePaddle's support for Kubernetes:

..  toctree::
  :maxdepth: 1

  k8s_en.md
  k8s_distributed_en.md

`OpenMPI <https://www.open-mpi.org>`_ is a mature high-performance parallel computing framework, which is widely used in the field of HPC. The following guide describes how to use OpenMPI to build PaddlePaddle's cluster training task:

..  toctree::
  :maxdepth: 1

  openmpi_en.md

`Fabric <http://www.fabfile.org>`_ is a convenient tool for program deployment and management. We provide a way to deploy and manage with Fabric. If you want to know more about it, please read the following guidelines:

..  toctree::
  :maxdepth: 1

  fabric_en.md

We also support the deployment of PaddlePaddle on AWS. Learn more about:

..  toctree::
  :maxdepth: 1

  k8s_aws_en.md

The examples can be found under `cluster_train_v2 <https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2>`_ .
