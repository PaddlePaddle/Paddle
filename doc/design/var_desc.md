## Background
PaddlePaddle divides the description of neural network computation graph into two stages: compile time and runtime.

PaddlePaddle use proto message to describe compile time graph for

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
  optional LoDTesnorDesc lod_tensor = 2; //
}
```

## Definition of LodTensorDesc

```proto
message LoDTensorDesc {
  enum Type {
    BOOL = 0;
    INT16 = 1;
    INT32 = 2;
    INT64 = 3;
    FP16 = 4;
    FP32 = 5;
    FP64 = 6
  }

  Type data_type = 1;
  repeated int dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
  optional int lod_level [default=0] = 3;
}
```

## Definition of Variable in Python

In Python API, layer will take Variable as Input, and return Variable as Output.

```python
image = Variable()
# fc1 and fc2 are both Variable
fc1 = layer.fc(input=image, output_size=10)
fc2 = layer.fc(input=fc1, output_size=20)
```

There should be a class `Variable` in python to help create and manage Variable.

```python
import VarDesc
import LoDTensorDesc
import framework

class Variable(object):
   def __init__(self, name, dims, type):
      self._name = name
      self.op = None
      tensor_desc = LoDTensorDesc(data_type=type, dims=dims)
      _var_desc = VarDesc(name=name, lod_tensor=tensor_desc)
      self._var = framework.CreateVar(_var_desc)

   def dims(self):
      return self._var.dims()

   def data_type(self):
       return self._var.data_type()
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
  W = Var(type=FP32, dims=[flatten_size(X, num_flatten_dims), output_size])
  b = Variable(type=FP32, dims=[output_size])
  out = Variable(type=FP32)
  y = operator.fc(X, W, b, output=out) # fc will put fc op input into out
  pd.InferShape(y)
  return out

x = var(dim=[-1, 640, 480])
y = layer.fc(x, output_size=100)
z = layer.fc(y, output_size=200)

paddle.train(z, ...)
print(y)
```
