#############################
Parameter Setting
#############################

..  contents::

1. Reduce Memory Consumption
-------------------

The training procedure of neural networks demands dozens of gigabytes of host memory or serval gigabytes of device memory, which is a rather memory consuming work. The memory consumed by PaddlePaddle framework mainly includes:
\:

* Cache memory for DataProvider (only on host memory),
* Memory for neurons' activation information (on both host memory and device memory),
* Memory for parameters (on both host memory and device memory),
* Other memory demands.

Other memory demands is mainly used to support the running demand of PaddlePaddle framework itself, such as string allocation，temporary variables, which are not considered currently.

Reduce DataProvider Cache Memory
++++++++++++++++++++++++++

PyDataProvider works under asynchronous mechanism, it loads together with the data fetch and shuffle procedure in host memory:

..  graphviz::

    digraph {
        rankdir=LR;
        Data Files -> Host Memory Pool -> PaddlePaddle Training
    }

Thus the reduction of the DataProvider cache memory can reduce memory occupancy, meanwhile speed up the data loading procedure before training. However, the size of the memory pool can actually affect the granularity of shuffle，which means a shuffle operation is needed before each data ﬁle reading process to ensure the randomness of data when try to reduce the size of the memory pool.

..  literalinclude:: src/reduce_min_pool_size.py

In this way, the memory consumption can be significantly reduced and hence the training procedure can be accelerated. More details are demonstrated in :ref:`api_pydataprovider2`.

The Neurons Activation Memory
++++++++++++++

Each neuron activation operating in a neural network training process contains certain amount of temporary data such as the activation data (like the output value of a neuron). These data will be used to update parameters in back propagation period. The scale of memory consumed by these data is mainly related with two parameters, which are batch size and the length of each Sequence. Therefore, the neurons activation memory consuming is actually in proportion to the information contains in each mini-batch training.

Two practical ways:

* Reduce batch size. Set a smaller value in network configuration settings(batch_size=1000) can be helpful. But setting batch size to a smaller value may affect the training result due to it is a super parameter of the neural network itself.
* Shorten the sequence length or cut oﬀ those excessively long sequences. For example, if the length of sequences in a dataset are mostly varies between 100 and 200, but there is sequence lengthen out to 10,000, then it’s quite potentially leads to OOM (out of memory), especially in RNN models such as LSTM.

The Parameters Memory
++++++++

The PaddlePaddle framework supports almost all popular optimizers. Different optimizers have different memory requirement. For example, the :code:`adadelta` consumes approximately 5 times memory

space than the weights parameter’s scale, which means the :code:`adadelta` needs at least :code:`500M` memory if the model ﬁle contains all

parameters needs :code:`100M`.

Some optimization algorithms such as :code:`momentum` are worth giving a shot.

2. Tricks To Speed Up Training
-------------------

The training procedure of PaddlePaddle may be speed up when considering following aspects:\：

* Reduce the time consumption of data loading
* Speed up training epochs
* Introduce more computing resources with the utilization of distribute training frameworks

Reduce The Time Consumption of Data Loading
++++++++++++++++++


The \ :code:`pydataprovider`\ holds big potential to speed up the data loading procedure if the cache pool and enable memory cache when use it. The principle of the reduction of :code:`DataProvider` cache pool is basically the same with the method which reduct the memory occupation with the set of a smaller cache pool.

..  literalinclude:: src/reduce_min_pool_size.py

Beside, the interface :code:`@provider` provides a parameter :code:`cache` to control cache. If set it to :code:`CacheType.CACHE_PASS_IN_MEM`, the data after the first :code:`pass` ( a pass means all data have be fed into the network for training) will be cached in memory and no new data will be read from the :code:`python` side in following :code:`pass` , instead from the cached data in memory. This strategy can also drop the time consuming in data loading process.


Accelerating Training Epochs
++++++++++++

Sparse training is supported in PaddlePaddle. The features needs to be trained is any of :code:`sparse_binary_vector`, :code:`sparse_vector` and :code:`integer_value` . Meanwhile, the Layer interacts with the training data need to turn the Parameter to sparse updating mode by setting :code:`sparse_update=True`.
Take :code:`word2vec` as an example, to train a language distance, one needs to predict the middle word with two words prior to it and next to it. The DataProvider of this task is:

..  literalinclude:: src/word2vec_dataprovider.py

The configuration of this task is:

..  literalinclude:: src/word2vec_config.py

Introduce More Computing Resources
++++++++++++++++++

More computing resources can be introduced with following manners:
* Single CPU platform training

  * Use multi-threading by set :code:`trainer_count`。

* Single GPU platform training

  * Set :code:`use_gpu` to train on single GPU.
  * Set :code:`use_gpu` and :code:`trainer_count` to enable multiple GPU training support.

* Cluster Training

  * Refer to :ref:`cluster_train` 。

3. Assign GPU Devices
------------------

Assume a computing platform consists of 4 GPUs which serial number from 0 to 3:

* Method1: specify a GPU as computing device by set:
 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_

..      code-block:: bash

        env CUDA_VISIBLE_DEVICES=2,3 paddle train --use_gpu=true --trainer_count=2

* Method2: Assign by —gpu_id:

..      code-block:: bash

        paddle train --use_gpu=true --trainer_count=2 --gpu_id=2


4. How to Fix Training Termination Caused By :code:`Floating point exception` During Training.
------------------------------------------------------------------------

Paddle binary catches floating exceptions during runtime, it will be terminated when NaN or Inf occurs. Floating exceptions are mostly caused by float overflow, divide by zero. There are three main reasons may raise such exception:

* Parameters or gradients during training are oversize, which leads to float overflow during calculation.
* The model failed to converge and diverges to a big value.
* Parameters may converge to a singular value due to bad training data. If the scale of input data is too big and contains millions of parameter values, float overflow error may arise when operating matrix multiplication.

Two ways to solve this problem:

1. Set :code:`gradient_clipping_threshold` as:

..  code-block:: python

    optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))

Details can refer to example `nmt_without_attention  <https://github.com/PaddlePaddle/models/blob/develop/nmt_without_attention/train.py#L35>`_ 示例。

2. Set :code:`error_clipping_threshold` as:

..  code-block:: python

    decoder_inputs = paddle.layer.fc(
        act=paddle.activation.Linear(),
        size=decoder_size * 3,
        bias_attr=False,
        input=[context, current_word],
        layer_attr=paddle.attr.ExtraLayerAttribute(
            error_clipping_threshold=100.0))

Details can refer to example `machine translation <https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/train.py#L66>`_ 。

The main difference between these two methods are:

1. They both block the gradient, but happen in different occasions，the former one happens when then :code:`optimzier` updates the network parameters while the latter happens when the back propagation computing of activation functions.
2. The block target are different, the former blocks the trainable parameters’ gradient while the later blocks the gradient to be propagated to prior layers.

Moreover, Such problems may be fixed with smaller learning rates or data normalization.

5.  Fetch Multi Layers’ Prediction Result With Infer Interface
-----------------------------------------------

* Join the layer to be used as :code:`output_layer` layer to the input parameters of  :code:`paddle.inference.Inference()` interface with:

..  code-block:: python

    inferer = paddle.inference.Inference(output_layer=[layer1, layer2], parameters=parameters)

* Assign certain ﬁelds to output. Take :code:`value` as example, it can be down with following code:

..  code-block:: python

    out = inferer.infer(input=data_batch, field=["value"])

It is important to note that:

* If 2 layers are assigned as output layer, then the output results consists of 2 matrixes.
* Assume the output of first layer A is a matrix sizes N1 * M1, the output of second layer B is a matrix sizes N2 * M2；
* By default, paddle.v2 will transverse join A and B, when N1 not equal to N2, it will raise following error:

..      code-block:: python

    ValueError: all the input array dimensions except for the concatenation axis must match exactly

The transverse of diﬀerent matrixes of multi layers mainly happens when:

* Output sequence layer and non sequence layer;
* Multiple output layers process multiple sequence with different length;

Such issue can be avoided by calling infer interface and set :code:`flatten_result=False`. Thus, the infer interface returns a python list, in which

* The number of elements equals to the number of output layers in the network;
* Each element in list is a result matrix of a layer, which type is numpy.ndarray;
* The height of each matrix outputted by each layer equals to the number of samples under non sequential mode or equals to the number of elements in the input sequence under sequential mode. Their width are both equal to the layer size in configuration.

6.  Fetch the Output of A Certain Layer During Training
-----------------------------------------------

In event_handler, the interface :code:`event.gm.getLayerOutputs("layer_name")` gives the forward output value organized in :code:`numpy.ndarray` corresponding to :code:`layer_name` in the mini-batch.
The output can be used in custom measurements in following way:

..      code-block:: python

        def score_diff(right_score, left_score):
            return np.average(np.abs(right_score - left_score))

        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 25 == 0:
                    diff = score_diff(
                        event.gm.getLayerOutputs("right_score")["right_score"][
                            "value"],
                        event.gm.getLayerOutputs("left_score")["left_score"][
                            "value"])
                    logger.info(("Pass %d Batch %d : Cost %.6f, "
                                "average absolute diff scores: %.6f") %
                                (event.pass_id, event.batch_id, event.cost, diff))

Note: this function can not get content of :code:`paddle.layer.recurrent_group` step, but output of  :code:`paddle.layer.recurrent_group` can be fetched.

7.  Fetch Parameters’ Weight and Gradient During Training
-----------------------------------------------

Under certain situations, knowing the weights of currently training mini-batch can provide more inceptions of many problems. Their value can be acquired by printing values in :code:`event_handler` (note that to gain such parameters when training on GPU, you should set :code:`paddle.event.EndForwardBackward`). Detailed code is as following:

..      code-block:: python

        ...
        parameters = paddle.parameters.create(cost)
        ...
        def event_handler(event):
            if isinstance(event, paddle.event.EndForwardBackward):
                if event.batch_id % 25 == 0:
                    for p in parameters.keys():
                        logger.info("Param %s, Grad %s",
                            parameters.get(p), parameters.get_grad(p))

Note that “acquire the output of a certain layer during training” or “acquire the weights and gradients of parameters during training ” both needs to copy training data from C++ environment to numpy, which have certain degree of inﬂuence on training performance. Don’t use these two functions when the training procedure cares about the performance.
