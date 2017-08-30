# Type (or Shape) Inference

## Starting from Examples

```python
x = layer.data("images", input_size=64*64)
y = layer.fc(x, output_size=100, activation=softmax)
l = layer.data("label", dims=10)
cost = layer.cross_entropy(y, l)
```

In this example, given the known size of input as 64*64, we can infer shapes of other variables:

1. x is a tensor of shape `[UNK, 64*64]`, where UNK is the minibatch size.
1. W created by layer.fc is a tensor of shape `[64*64, 100]`
1. b created by layer.fc is a tensor `[100]`
1. x*W is an intermediate tensor of shape `[UNK, 100]`
1. x*W+b is an intermediate tensor of shape `[UNK, 100]`
1. y = softmax(x*W+b) is a tensor of shape `[UNK, 100]`
1. l is a tensor of shape `[UNK, 10]`
1. cost is a tensor of shape `[UNK, 1]`

It is possible that we have variable-length inputs, e.g., text of words:

```python
x = layer.data("paragraph", lod_level=2, input_size=6000)
```

`lod_level` specifies that a paragraph is composed of sentences, which consists of words.  `input_size` is the dictionary size -- we have 6000 known words.  In this case,

- x is an LODTensor of shape `[UNK, UNK, 6000], lod_level=2`.

If the input is a video, or a sequence of images:

```python
x = layer.data("video", lod_level=1, input_size=640*480)
```

- x is an LODTensor of shape `[UNK, UNK, 640*480], lod_level=1`.

## VarDesc

```protobuf
message VarDesc {
  enum Type {
    INT = 0;
    FLOAT = 1;
    STRING = 2;
    INTS = 3;
    FLOATS = 4;
    STRINGS = 5;
    LOD_TENSOR = 6;
  }
  Type type = 1;
  optional LodTesnorDesc lod_tensor = 2; // when type==LOD_TENSOR
}

message LoDTensorDesc {
  optional int lod_level [default=0] = 3;
  repeated int dims = 1; // [UNK, UNK, 6000] is saved as [-1, -1, 6000]
  enum Type {
    INT8 = 0;
    INT16 = 1;
    INT32 = 2;
    INT64 = 3;
    FP16 = 4;
    FP32 = 5;
  }
  Type element_type = 2;
}
```

Note: VarDesc is not OpDesc::Var.

## The Lifecycle of VarDesc

In C++ class `Block` (used to known as `NetOp`):

```c++
class Block : public OperatorBase {
 private:
  map<string, VarDesc> vars_;
  vector<OperatorBase*> ops_;
};
```

```c++
class OperatorBase {
 private:
  const OpDesc& desc_;
}
```

## InferShape


```c++
class OperatorBase {
 public:
  OperatorBase(const std::string& type, const VariableNameMap& inputs,
               const VariableNameMap& outputs, const AttributeMap& attrs,
               Block* block = nullptr) {
    ...
    InferShape(block);
  }
 private:
  virtual InferShape(Block* block) = 0;
};
```



## Block as a proto message

```protobuf
message BlockDesc {
  repeated VarDesc vars = 1;
  repeated OperatorDesc ops = 2;
}

message OperatorDesc {
  required string type = 1;
  repeated string inputs = 2;
  repeated string outputs = 3;
  repeated AttrDesc attrs = 4;
}

message VarDesc {
  required name string = 1;
  enum Type { .. }
  Type type = 2;
  optional LoDTensorDesc lod_tensor = 3;
}

message AttrDesc {
    required string name = 1;
    required AttrType type = 2;
    optional int32 i = 3;
    optional float f = 4;
    optional string s = 5;
    repeated int32 ints = 6;
    repeated float floats = 7;
    repeated string strings = 8;
    optional BlockDesc block = 9;
};

enum AttrType {
  INT = 0;
  FLOAT = 1;
  STRING = 2;
  INTS = 3;
  FLOATS = 4;
  STRINGS = 5;
  BLOCK = 6;
}
```

```python
a = Var()
b = Var()
c = layer.rnn(
      step_net={
