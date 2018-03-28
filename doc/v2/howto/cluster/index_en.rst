Distributed Training
====================

The effectiveness of the deep learning model is often directly related to the scale of the data: it can generally achieve better results after increasing the size of the dataset on the same model. However, it can not fit in one single computer when the amount of data increases to a certain extent. At this point, using multiple computers for distributed training is a natural solution. In distributed training, the training data is divided into multiple copies (sharding), and multiple machines participating in the training read their own data for training and collaboratively update the parameters of the overall model.

Distributed training generally has framwork as shown below:

.. image:: src/ps_en.png
   :width: 500

- Data shard: training data will be split into multiple partitions, trainers use the partitions of the whole dataset to do the training job.
- Trainer: each trainer reads the data shard, and train the neural network. Then the trainer will upload calculated "gradients" to parameter servers, and wait for parameters to be optimized on the parameter server side. When that finishes, the trainer download optimized parameters and continues its training.
- Parameter server: every parameter server stores part of the whole neural network model data. They will do optimization calculations when gradients are uploaded from trainers, and then send updated parameters to trainers.

The training of synchronous random gradient descent for neural network can be achieved by cooperation of trainers and parameter servers.

PaddlePaddle supports both synchronize stochastic gradient descent (SGD) and asynchronous SGD.

Before starting the cluster training, you need to prepare the cluster configuration, PaddlePaddle installation, and other preparations. To understand how to configure the basic environment for distributed training, check the link below:

..  toctree::
  :maxdepth: 1

  preparations_en.md

Cluster training has a large number of configurable parameters, such as the number of machines used, communication ports, etc. To learn how to configure the distributed training process by setting startup these parameters, check the link below:

..  toctree::
  :maxdepth: 1

  cmd_argument_en.md

PaddlePaddle is compatible with a variety of different clusters. Each cluster has its own advantages, To learn how to run PaddlePaddle in different types of them, check the link below:

..  toctree::
  :maxdepth: 1

  multi_cluster/index_en.rst
