在不同集群中运行
================
用户的集群环境不尽相同，为了方便大家的部署，我们提供了多种的集群部署方式，方便提交集群训练任务，以下将一一介绍:

`Kubernetes <http://kubernetes.io>`_ 是Google开源的容器集群的调度框架，支持大规模集群生产环境的完整集群方案。以下指南展示了PaddlePaddle对Kubernetes的支持：

..  toctree::
  :maxdepth: 1

  k8s_cn.md
  k8s_distributed_cn.md

`OpenMPI <https://www.open-mpi.org>`_  是成熟的高性能并行计算框架，在HPC领域使用非常的广泛。以下指南介绍了如何使用OpenMPI来搭建PaddlePaddle的集群训练任务:

..  toctree::
  :maxdepth: 1

  openmpi_cn.md

`Fabric <http://www.fabfile.org>`_ 是一个方便的程序部署和管理工具。我们提供了使用Fabric 进行部署、管理的方法，如果想详细了解，请阅读以下指南:

..  toctree::
  :maxdepth: 1

  fabric_cn.md

我们也支持在AWS上部署PaddlePaddle，详细请了解:

..  toctree::
  :maxdepth: 1

  k8s_aws_cn.md

您可以在 `cluster_train_v2 <https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2>`_ 找到以上相关的例子。

