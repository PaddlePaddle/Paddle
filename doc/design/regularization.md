# Regularization in PaddlePaddle

## Introduction to Regularization
A central problem in machine learning is how to design an algorithm that will perform well not just on the training data, but also on new data. Many strategies are used by machine learning practitioners to reduce the test error, possibly at the expense of increased training error. These strategies are collectively known as **regularization**. 

### Parameter Norm Penalties
Most common regularization approaches in deep learning are based on limiting the capacity of the models by adding a parameter norm penalty to the objective function `J`. This is given as follows:


![](http://bit.ly/2zt9OJY)

The parameter `alpha` is a hyperparameter that weights the relative contribution of the norm penalty term, `omega`, relative to the standard objective function `J`.

The most commonly used norm penalties are the L2 norm penalty and the L1 norm penalty. These are given as follows:

##### L2 Regularization:
![](http://bit.ly/2yo9Rcp)

##### L1 Regularization
![](http://bit.ly/2zu4shg)

A much more detailed mathematical background of reguilarization can be found [here](http://www.deeplearningbook.org/contents/regularization.html).


## How to do Regularization in PaddlePaddle

On surveying existing frameworks like Tensorflow, PyTorch, Caffe, etc, it can be seen that there are 2 common approaches of doing regularization:

1. Making regularization a part of the optimizer using an attribute like `weight_decay` that is used to control the scale of the L2 Penalty. This approach is used in PyTorch as follows:
	```python
	opt =  torch.optim.SGD(params, lr=0.2, weight_decay=0.2)
	```
    At every optimization step, this code will add the gradient of the L2 Norm of the params to the gradient of the params with respect to the loss function. This can seen in the following code snippet:
    ```python
    if weight_decay != 0:
    	d_p.add_(weight_decay, p.data)
    ```
    This is a very restyrictive way of doing regularization and does not give the users enough flexibility. 
    
    **Advantages**:
    -  It is easy to implement for us.
    -  Faster execution of backward. However, it can be done manually by advanced users too.

	**Disadvantages**:
    - Not flexible for other regularizations such as L1/L0 regularization.
    - Does not allow for different regularization coefficient for different parameters. For example, in most models, ony the weight matrices are regularized and the bias vectors are unregularized.
    - Tightly coupled optimizer and regularization implementation. 


2. Adding regularization ops to the graph through Python API. This approach is used by Tensorflow and Caffe. Using this approach, we manually add regularization ops to the graph and then add the regularization loss to the final loss function before sending them to the optimizer.

	**Advantages**:
    - Allows for greater flexibility to the users of Paddle. Using this approach, the users can put different regularization to different parameters and also choose parameters that are not a part of regularization.
    - Makes it easy for the users to customize and extend the framework. 

	**Disadvantages**:
    - Implementation requires comprehensive design and time. 

## Proposal for Regularization in PaddlePaddle

### Low-Level implementation

In the new design, we propose to create new operations for regularization. For now, we can add 2 ops thgat correspond to the most frequently used regularizations:
- L2_regularization_op
- L1_regularization_op

These ops can be like any other ops with their own CPU/GPU implementations either using Eigen or separate Cpu and GPU kernels. As the initial implementation, we can implement their kernels using Eigen following the abstraction pattern implemented for [Activation Ops](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/accuracy_op.h). This abstraction pattern can make it very easy to implement new regularization schemes. other than L1 and L2 norm penalties. 

The idea of building ops for regularization is in sync with the refactored Paddle philosophy of using operators to represent any computation unit. The way these ops will be added to the computation graph, will be decided by the [layer functions](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/python_api.md#layer-function) in Python API. 


### Python API implementation for Regularization

Using the low level ops, `L2_regularization_op` and `L1_regularization_op`, any user can add regularization to their computation graphs. However, this will require a lot of lines of code and we should design Python APIs that support regularization. An example of such an API can be seen in [Keras](https://keras.io/regularizers/). As per the PaddlePaddle [Python API design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/python_api.md), the layer functions are responsible for creating operators, operator parameters and variables. Since regularization is a property of parameters, it makes sense to create these in the layer functions. 

#### Creation of Regularization ops
There are two possibilities for creating the regularization ops:
1. We create these ops immediately while building the computation graph. 
2. We add these ops in a lazy manner, just before the backward, similar to the way the optimization ops are added. 

The proposal is to add these ops in a lazy manner just before the backward pass. 

#### Storage of Regularization attributes

Since we want to create the regularization ops in a lazy manner, the regularization attributes (type of regularization and weight of regularization penalty) can be stored as attributes of the [`Parameter`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/framework/framework.py#L421) class. This is because regularization is a property of the parameters and storing regularization properties with Parameters also allows for shared parameters. 

#### High-level API

In PaddlePaddle Python API, users will primarily rely on [layer functions](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/python_api.md#layer-function) to create neural network layers. Hence, we lso need to provide regularization functionality in layer functions. The design of these APIs can be postponed for later right now. A good reference for these APIs can be found in [Keras](https://keras.io/regularizers/) and also by looking at Tensorflow in [`tf.contrib.layers`](https://www.tensorflow.org/api_guides/python/contrib.layers).





    
