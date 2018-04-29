############
Basic use concept
############

PaddlePaddle is a deep learning platform derived from Baidu. PaddlePaddle provides a rich API for deep learning researchers. It can easily perform tasks such as neural network configuration and model training. Here we will introduce the basic concept of PaddlePaddle, and show how to use PaddlePaddle to solve a classic linear regression problem. Before using this document, please refer to the `installation documentation <../../build_and_install/index_cn.html>`_ to complete the installation of PaddlePaddle. 


Network Configuration
============

Load PaddlePaddle
----------------------

Before configuring the network, you first need to load the corresponding Python library and perform initialization.

..	code-block:: bash

    import paddle.v2 as paddle
    import numpy as np
    paddle.init(use_gpu=False)


Build the neural network
-----------------------

Building a neural network is like using a building block to build a pagoda. In PaddlePaddle, the layer is our building block, and the neural network is the pagoda we want to build. We use different layers to combine to build a neural network.The bottom of the pagoda needs a solid base to support it, similarly, the neural network needs some specific layers as input interfaces to complete the network training.

For example, we can define the following layer to describe the neural network input:

..	code-block:: bash

    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

Where x indicates that the input data is a Two-dimensional vector, and y indicates that the input data is a dense One-dimensional vector.

PaddlePaddle supports different types of input data, mainly including four types, and three sequence modes.

Four data types:

* dense_vector: Dense floating point vector.
* sparse_binary_vector: Sparse 01 vector, that is, most of the value is 0, the other must be 1.
* sparse_float_vector: Sparse vectors, most of which are 0, others can be any floating-point number.
* integer: Integer label.

Three sequence modes:

* SequenceType.NO_SEQUENCE：Not a sequence.
* SequenceType.SEQUENCE：Is a time sequence.
* SequenceType.SUB_SEQUENCE： It is a time sequence, and each element of the sequence is also a time sequence.

Different data types and sequence modes return different formats. The list is as follows:

+----------------------+---------------------+-----------------------------------+------------------------------------------------+
|                      | NO_SEQUENCE         | SEQUENCE                          |  SUB_SEQUENCE                                  |
+======================+=====================+===================================+================================================+
| dense_vector         | [f, f, ...]         | [[f, ...], [f, ...], ...]         | [[[f, ...], ...], [[f, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_binary_vector | [i, i, ...]         | [[i, ...], [i, ...], ...]         | [[[i, ...], ...], [[i, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_float_vector  | [(i,f), (i,f), ...] | [[(i,f), ...], [(i,f), ...], ...] | [[[(i,f), ...], ...], [[(i,f), ...], ...],...] |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| integer_value        |  i                  | [i, i, ...]                       | [[i, ...], [i, ...], ...]                      |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+

Where f represents a floating point number and i represents an integer.

Note: For sparse_binary_vector and sparse_float_vector, PaddlePaddle stores the index with the value position. E.g,

- For a 5-dimensional non-sequential sparse 01 vector ``[0, 1, 1, 0, 0]`` ，type is sparse_binary_vector，and the return is ``[1, 2]`` 。
- For a 5-dimensional non-sequential sparse floating-point vector ``[0, 0.5, 0.7, 0, 0]`` ，type is sparse_float_vector，and the return is ``[(1, 0.5), (2, 0.7)]`` 。


After defining the input layer, we can use other layers to combine. When combining, we need to specify the input source of the layer.

For example, we can define the following layer combination:

..	code-block:: bash

    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    cost = paddle.layer.square_error_cost(input=y_predict, label=y)

Among them, x and y are the input layer described before; `y_predict` is to receive x as an input, followed by a fully connected layer; `cost` receives y_predict and y as input, and is connected to a squared error layer.

The last layer of cost records all the topological structures of the neural network. By combining different layers, we can complete the construction of the neural network.


Training model
============

After completing the construction of the neural network, we first need to create parameters that need to be optimized according to the neural network structure, and create an optimizer.
After that, we can create a trainer to train the network.

..	code-block:: bash

    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(momentum=0)
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

The trainer receives three parameters, including neural network topology, neural network parameters, and iterative equations.

In the process of building a neural network, we only describe the input to the neural network. The trainer needs to read the training data for training. The PaddlePaddle uses the reader to load the data.

..	code-block:: bash

    # define training dataset reader
    def train_reader():
        train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
        train_y = np.array([[-2], [-3], [-7], [-7]])
        def reader():
            for i in xrange(train_y.shape[0]):
                yield train_x[i], train_y[i]
        return reader

Finally we can start the training by calling trainer's train method:

..	code-block:: bash

    # define feeding map
    feeding = {'x': 0, 'y': 1}

    # event_handler to print training info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
    # training
    trainer.train(
        reader=paddle.batch(train_reader(), batch_size=1),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=100)

For more information on using the PaddlePaddle, please refer to `Advanced guide <../../howto/index_cn.html>`_。

Linear regression complete example
==============

Here is an example of using linear regression to fit a straight line in three-dimensional space:

..  literalinclude:: src/train.py
    :linenos:

Use the above trained model to predict, take one of the models params_pass_90.tar, input the vector group that needs to be predicted, and print the output:

..  literalinclude:: src/infer.py
    :linenos:

For practical application of linear regression, refer to PaddlePaddle book `the first chapter <http://book.paddlepaddle.org/index.html>`_。
