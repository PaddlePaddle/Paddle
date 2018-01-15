## Background
PaddlePaddle divides the description of neural network computation into two stages: compile time and runtime. At compile time, the neural network computation is described as a `ProgramDesc` whereas at runtime an `Executor` interprets the `ProgramDesc` to compute the operations.

PaddlePaddle use proto message to describe compile time program because

1. The computation program description must be serializable and saved in a file.
1. During distributed training, the sreialized program will be sent to multiple workers. It should also be possible to break the program into different components, each of which can be executed on different workers.

The computation `Program` consists of nested `Blocks`. Each `Block` will consist of data(i.e. `Variable`)  and  `Operations`. The concept to represent them is in the table below.

| |compile time|runtime|
|---|---|---|
|Data|VarDesc(proto)|Variable(cpp)|
|Operation|OpDesc(proto)|Operator(cpp)|


## Definition of VarDesc

A VarDesc should have a name, and value. The are two kinds of variable type in compile time, they are `LoDTensor` and `SelectedRows`. 

```proto
message VarDesc {
  required string name = 1;
  enum VarType {
    LOD_TENSOR = 0;
    SELECTED_ROWS = 1;
  }
  required VarType type = 2;
  optional LoDTensorDesc lod_desc = 3;
  optional TensorDesc selected_rows_desc = 4;
  optional bool persistable = 5 [ default = false ];
}
```

## Definition of TensorDesc

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

message TensorDesc {
  required DataType data_type = 1;
  repeated int64 dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
}
```

A TensorDesc describes `SelectedRows` and `LoDTensor`. For details of `SelectedRows`, please reference [`SelectedRows`](./selected_rows.md).

## Definition of LodTensorDesc

```proto
message LoDTensorDesc {
  required TensorDesc tensor = 1;
  optional int lod_level = 2;
}
```

A LoDTensorDesc contains a tensor and a lod_level.

## Definition of Variable in Python

For Variable in Python, please reference [`Python API`](./python_api.md).
