# Design Doc: From V2 Python API to Refactorized C++ Core

## Background

We released V2 Python API in February 2017.  This new API would allow users 

1. to write concise programs, and
1. to write pure Python programs that call PaddlePaddle's C++ core as a dynamic library.
   
However, the C++ core code has been old and accumulated messy parts in the recent four years.  Also, the core doesn't implement the key concepts exposed through the V2 Python API.  Therefore, we need to rewrite the C++ core and make sure that it works with the API.

## The V2 API

The design doc of the API is at [here](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/api.md).  For a quick revisit, let's start with a simple program:

```python
paddle.init(use_gpu=False, trainer_count=1)

x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
cost = paddle.layer.mse(input=y_predict, label=y)

parameters = paddle.parameters.create(cost)
optimizer = paddle.optimizer.Momentum(momentum=0)
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)
```

## The New C++ Core

The new C++ core implements the following key concepts:

1. Operator, whose inputs and outputs are variables.
1. Variable, each can hold a value of specified type.
1. Tensor, a value commonly hold by Variable.
1. Scope, keep Variables.

## Fill the Gap

We can see the gap between the Python API and the C++ core:

1. the API describes a network layer-by-layer, however, 
1. the core runs a graph of operators connected by variables in scopes.

We'd expose the core's low-level API to Python via [`pybind11`](https://github.com/pybind/pybind11), and write the high-level API `paddle.layer.xxx` to call the low-level API.

## From Layer to Operator and Variables

 As can see from above example, the input of a layer is always a layer -- the very beginning is often a `paddle.layer.data`.
 
 Above example identifies a network by its final layer.  For join cost, we would have a `paddle.layer.join`.
 
 So let's think the three kinds of layers:
 
1. `paddle.layer.data` should create and return a Variable labeled "input".  `paddle.trainer.SGD` will assign a mini-batch to this Variable, before calling `Operator::Run` of operators in the topological order.
        
1. `paddle.layer.fc` and other computational layers should create and return their output variables.  They should also create operators and model parameter variables, which should be labeled "parameter".
   
1. `paddle.layer.mse` and optimization objective layers work as computational layers. As they return the created output variables, and their return value is used by `paddle.paramters.create` and `paddle.trainer.SGD` to identify the network, we must be able to trace the network behind these variables.
   
## Proposals

1. class `Scope` doesn't need to maintain a 

   ```cpp
   map<string/*name*/, shared_ptr<Variable>> vars_;
   ```
   
   because the name of a variable is just an ID that doesn't have to be human readable, thus can be replaced by the address of the Variable object.  `Scope` could be
   
   ```cpp
   vector<shared_ptr<Variable>> vars_;
   ```
   
1. `paddle.layer.X` should be able to create Variables.  They can call `Scope::NewVar` and gets the address of the created Variables.
   
1. A Variable should maintain a label of its purpose like "input", "objective", "parameter", so that `paddle.parameters.create` can enlist those with label "parameter".
   
1. A Variable should maintain two lists

   ```cpp
   vector<shared_ptr<Operator>> readers_, writers_;
   ```
   
   For example, the `operator::Mul` created by `paddle.layer.fc` should be a reader of the variable created by `paddle.layer.data` and the writer of the output variable created by `paddle.layer.fc`.
