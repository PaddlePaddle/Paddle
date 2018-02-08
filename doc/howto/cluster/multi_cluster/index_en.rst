Use different clusters
======================

PaddlePaddle supports running jobs on several platforms including:
- `Kubernetes <http://kubernetes.io>`_ open-source system for automating deployment, scaling, and management of containerized applications from Google.
- `OpenMPI <https://www.open-mpi.org>`_ Mature high performance parallel computing framework.
- `Fabric <http://www.fabfile.org>`_ A cluster management tool. Write scripts to submit jobs or manage the cluster.

We'll introduce cluster job management on these platforms. The examples can be found under `cluster_train_v2 <https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2>`_ .

These cluster platforms provide API or environment variables for training processes, when the job is dispatched to different nodes. Like node ID, IP or total number of nodes etc.

..  toctree::
  :maxdepth: 1

  fabric_en.md
  openmpi_en.md
  k8s_en.md
  k8s_aws_en.md
