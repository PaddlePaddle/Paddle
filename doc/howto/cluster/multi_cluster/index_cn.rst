在不同集群中运行
================

PaddlePaddle可以使用多种分布式计算平台构建分布式计算任务，包括：
- `Kubernetes <http://kubernetes.io>`_ Google开源的容器集群的调度框架，支持大规模集群生产环境的完整集群方案。
- `OpenMPI <https://www.open-mpi.org>`_ 成熟的高性能并行计算框架。
- `Fabric <http://www.fabfile.org>`_ 集群管理工具。可以使用`Fabric`编写集群任务提交和管理脚本。

对于不同的集群平台，会分别介绍集群作业的启动和停止方法。这些例子都可以在 `cluster_train_v2 <https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2>`_ 找到。

在使用分布式计算平台进行训练时，任务被调度在集群中时，分布式计算平台通常会通过API或者环境变量提供任务运行需要的参数，比如节点的ID、IP和任务节点个数等。

..  toctree::
  :maxdepth: 1

  fabric_cn.md
  openmpi_cn.md
  k8s_cn.md
  k8s_distributed_cn.md
  k8s_aws_cn.md
