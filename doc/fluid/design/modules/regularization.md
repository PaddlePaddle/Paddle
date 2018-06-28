# Regularization in PaddlePaddle

## Introduction to Regularization
A central problem in machine learning is how to design an algorithm that will perform well not just on the training data, but also on new data. A frequently faced problem is the problem of **overfitting**, where the model does not make reliable predictions on new unseen data. **Regularization** is the process of introducing additional information in order to prevent overfitting. This is usually done by adding extra penalties to the loss function that restricts the parameter spaces that an optimization algorithm can explore.

### Parameter Norm Penalties
Most common regularization approaches in deep learning are based on limiting the capacity of the models by adding a parameter norm penalty to the objective function `J`. This is given as follows:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/loss_equation.png" align="center"/><br/>

The parameter `alpha` is a hyperparameter that weights the relative contribution of the norm penalty term, `omega`, relative to the standard objective function `J`.

The most commonly used norm penalties are the L2 norm penalty and the L1 norm penalty. These are given as follows:

##### L2 Regularization:
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/l2_regularization.png" align="center"/><br/>

##### L1 Regularization
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/l1_regularization.png" align="center"/><br/>

A much more detailed mathematical background of regularization can be found [here](http://www.deeplearningbook.org/contents/regularization.html).

## Regularization Survey

A detailed survey of regularization in various deep learning frameworks can be found [here](https://github.com/PaddlePaddle/Paddle/wiki/Regularization-Survey).

## Proposal for Regularization in PaddlePaddle

### Low-Level implementation

In the new design, we propose to create new operations for regularization. For now, we can add 2 ops that correspond to the most frequently used regularizations:
- L2_regularization_op
- L1_regularization_op

These ops can be like any other ops with their own CPU/GPU implementations either using Eigen or separate CPU and GPU kernels. As the initial implementation, we can implement their kernels using Eigen following the abstraction pattern implemented for [Activation Ops](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/accuracy_op.h). This abstraction pattern can make it very easy to implement new regularization schemes other than L1 and L2 norm penalties.

The idea of building ops for regularization is in sync with the refactored Paddle philosophy of using operators to represent any computation unit. The way these ops will be added to the computation graph, will be decided by the [layer functions](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#layer-function) in Python API.

### Computation Graph

Below is an example of a really simple feed forward neural network.

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/feed_forward.png" align="center"/><br/>

The Python API will modify this computation graph to add regularization operators. The modified computation graph will look as follows:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/feed_forward_regularized.png" align="center"/><br/>
   
### Python API implementation for Regularization

Using the low level ops, `L2_regularization_op` and `L1_regularization_op`, any user can add regularization to their computation graphs. However, this will require a lot of lines of code and we should design Python APIs that support regularization. An example of such an API can be seen in [Keras](https://keras.io/regularizers/). As per the PaddlePaddle [Python API design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md), the layer functions are responsible for creating operators, operator parameters and variables. Since regularization is a property of parameters, it makes sense to create these in the layer functions.

#### Creation of Regularization ops
There are two possibilities for creating the regularization ops:
1. We create these ops immediately while building the computation graph.
2. We add these ops in a lazy manner, just before the backward, similar to the way the optimization ops are added.

The proposal is to add these ops in a lazy manner just before the backward pass.

#### Storage of Regularization attributes

Since we want to create the regularization ops in a lazy manner, the regularization attributes (type of regularization and weight of regularization penalty) can be stored as attributes of the [`Parameter`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/framework/framework.py#L421) class. This is because regularization is a property of the parameters and storing regularization properties with Parameters also allows for shared parameters.

#### High-level API

In PaddlePaddle Python API, users will primarily rely on [layer functions](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#layer-function) to create neural network layers. Hence, we also need to provide regularization functionality in layer functions. The design of these APIs can be postponed for later right now. A good reference for these APIs can be found in [Keras](https://keras.io/regularizers/) and also by looking at Tensorflow in [`tf.contrib.layers`](https://www.tensorflow.org/api_guides/python/contrib.layers).
