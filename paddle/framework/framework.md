# Design Doc: Framework

This design is based on a group discussion and @reyoung's Gist https://gist.github.com/reyoung/dd1441659ef4806e0a1368f6fb8c6211.

This design includes the following aspects:

1. Define operators' protocol (how operators connect with each other to form a neural network) in C++.
2. Create (automatically, if possible) a Python function for each operator that users can call to build a neural network for training/inference.
3. When users call above Python functions to describe a network, this network should be able to serialized, so could be loaded for further processing, e.g., passed to TensorRT.
4. The described neural network should have a C++ in-memory data structure which can be trained or used for inference.

## Key Concepts

For 1. and 2., we need C++ source code to define the protocol, which is similar to TensorFlow's Op, and the definition should be saved into protobuf messgaes, wihch can then be loaded by Python code that automaticlally generates Python *operator creation functions*.  We'd name these protobuf message with suffix `Proto`, e.g., `OperatorProto`. we'd name the Python code that generates other Python functions `create_python_op_creators`.

For 3. and 4., we need another set of protobuf messages, namely `OperatorDesc`, which describes *operator instances*.  Python op creation functions should fill in `OperatorDesc` and use it as the parameter in the calling of a  C/C++ function that creates the operator instance in C++'s in-memory data structure.

In summary, we have the following concepts:

1. *operator definition*, which is a C++ class derived from `paddle::framework::Operator`.
1. *operator protocol*, which is the interface of each operator, recorded in protobuf message `paddle::framework::OperatorProto` defined as a C++ global variable.
1. *create_python_op_creators*, which is a Python function that reads operator protocols from above global variable and creates the operator creators,
1. *operator creators*, which are Python functions, each when called fills its arguments into a `OperatorDesc` protobuf message and passes it to C++ function `paddle::framework::CreateOperatorInstance`.

## Comparisons

In TensorFlow's terminology, the operator protocol is known as *Op* and one or more C++ classes that implements this protocol are called *Kernels*.  In PaddlePaddle, we have the protocol, but we are not expecting multiple kernels for each op, because we can always configure an operator to take various actions, like calling MKL or CUDNN to do the computation.  This similar to Caffe2, which doesn't differentiate Ops and Kernels as well. 

Caffe2 has one set of protobuf messages, that similar to `OperatorProto`, because it doesn't automatically generates Python operator creators.  PaddlePaddle has two: `OperatorProto` an `OperatorDesc`.


## `OperatorProto` and Network Configuration

An example usage of the generated operator creator is as follows:

```python
HOUSE_FEATURE_NUM=30
nn = paddle.op.mse(
  paddle.op.fc(
    input = paddle.op.data(name="house_features"),
    weights = paddle.framework.Tensor(30, 10),
    bias = paddle.framework.Tensor(1, 10),
    activator = paddle.op.LeRU),
  paddle.op.data(name="house_price"))

```

```protobuf
message OperatorProto {
  repeated 
```
