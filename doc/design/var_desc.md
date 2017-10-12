## Background
PaddlePaddle divides the description of neural network computation graph into two stages: compile time and runtime.

PaddlePaddle use proto message to describe compile time graph because

1. Computation graph should be able to be saved to a file.
1. In distributed training, the graph will be serialized and send to multiple workers.

The computation graph is constructed by Data Node and Operation Node. The concept to represent them is in the table below.

| |compile time|runtime|
|---|---|---|
|Data|VarDesc(proto)|Variable(cpp)|
|Operation|OpDesc(proto)|Operator(cpp)|


## Definition of VarDesc

A VarDesc should have a name and value, in PaddlePaddle, the value will always be a tensor. Since we use LoDTensor most of the time. We add a LoDTesnorDesc to represent it.

```proto
message VarDesc {
  required string name = 1;
  optional LoDTensorDesc lod_tensor = 2;
}
```

## Definition of LodTensorDesc

```proto
enum DataType {
  BOOL = 0;
  INT16 = 1;
  INT32 = 2;
  INT64 = 3;
  FP16 = 4;
  FP32 = 5;
  FP64 = 6;
}

message LoDTensorDesc {
  required DataType data_type = 1;
  repeated int32 dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
  optional int32 lod_level = 3 [default=0];
}
```

## Definition of Variable in Python

In Python API, layer will take Variable as Input, and return Variable as Output. There should be a class `Variable` in python to help create and manage Variable.

```python
image = Variable(dims=[-1, 640, 480])
# fc1 and fc2 are both Variable
fc1 = layer.fc(input=image, output_size=10)
fc2 = layer.fc(input=fc1, output_size=20)
```
### what should class `Variable` Have
1. `name`.a name of string type is used to mark the value of the Variable.
1. `initializer`. Since our Tensor does not have value. we will always use some Operator to fullfill it when run. So we should have a initialize method to help add the init operator.
1. `operator`. Variable should record which operator produce itself. The reaon is:
  - we use pd.eval(targets=[var1, var2]) to run the related ops to get the value of var1 and var2. var.op is used to trace the dependency of the current variable.

In PaddlePaddle, we use Block to describe Computation Graph, so in the code we will use Block but not Graph.

```python
import VarDesc
import LoDTensorDesc
import framework

def AddInitialOperator(variable, initializer):
	# add an initialize Operator to block to init this Variable

class Variable(object):
   def __init__(self, name, dims, type, initializer):
      self._block = get_default_block()
      self._name = name
      self.op = None

      tensor_desc = LoDTensorDesc(data_type=type, dims=dims)
      _var_desc = VarDesc(name=name, lod_tensor=tensor_desc)
      self._var = framework.CreateVar(_var_desc)
      self._block.add_var(self)

      # add initial op according to initializer
      if initializer is not None:
          AddInitialOperator(self, initializer)

   def dims(self):
      return self._var.dims()

   def data_type(self):
       return self._var.data_type()

   def to_proto(self):
       pass
```

Then we can use this Variable to create a fc layer in Python.

```python
import paddle as pd

def flatten_size(X, num_flatten_dims):
  prod = 1 # of last num_flatten_dims
  for i in xrange(num_flatten_dims):
    prod = prod * X.dims[-i-1]
  return prod

def layer.fc(X, output_size, num_flatten_dims):
  W = Variable(pd.random_uniform(), type=FP32, dims=[flatten_size(X, num_flatten_dims), output_size])
  b = Variable(pd.random_uniform(), type=FP32, dims=[output_size])
  out = Variable(type=FP32)
  y = operator.fc(X, W, b, output=out) # fc will put fc op input into out
  pd.InferShape(y)
  return out

x = Variable(dims=[-1, 640, 480])
y = layer.fc(x, output_size=100)
z = layer.fc(y, output_size=200)

paddle.eval(targets=[z], ...)
print(z)
```
