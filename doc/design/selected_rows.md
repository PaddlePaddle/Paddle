# Design Doc: Selected Rows

`SelectedRows` is a kind of sparse tensor data type, which is designed to support `embedding` operators. The gradient of embedding table is a sparse tensor. Only a few rows are non-zero values in that tensor. It is straightforward to represent the sparse tensor by the following sparse tensor data structure:

```cpp
class SelectedRows {
 private:
  vector<int> rows_;
  Tensor value_;
  int height_;
};
```

The field `height_` shows the first dimension of `SelectedRows`. The `rows` are the indices of which rows of `SelectedRows` are non-zeros. The `value_` field is an N-dim tensor and shape is `[rows.size() /* NUM_ROWS */, ...]`, which supplies values for each row. The dimension of `SelectedRows` satisfies `[height_] + value_.shape[1:]`.

For example, given `height_=100`, `rows_ = [73, 84]`, the `value_ = [[1.0, 2.0], [3.0, 4.0]]` specifies that it is a `100*2` matrix, the 73rd row of that sparse tensor is `[1.0, 2.0]`, the 84th row of that sparse tensor is `[3.0, 4.0]`.


## SelectedRows in Protobuf

`SelectedRows` is a kind of `Variable`. `VarDesc` in protobuf should describe the `SelectedRows` information. Only the tensor dimension of a `SelectedRows` will be described in compile-time since the `rows_` and `value_` are related to training data. The `VarDesc` will unify `Dimension` field since `Dimension` is a attribute of both `LoDTensor` and `SelectedRows`.

```proto
message TensorDesc {
  required DataType data_type = 1;
  repeated int64 dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
}

message VarDesc {
  required string name = 1;
  enum VarType { 
    LoDTensor = 0;
    SelectedRows = 1;
  }
  optional VarType type = 2 [ default = LoDTensor ];
  optional TensorDesc tensor = 3;
  optional int32 lod_level = 4 [ default = 0 ];
  optional bool persistable = 5 [ default = false ];
}
```

## InferShape for Selected Rows

Just like `LoD` information, `InferShape` method will inference output tensor type as well. The operator should decide whether its output is a `SelectedRows` or `Dense` tensor.

For example, the gradient operator of `TableLookup` will always generate `SelectedRows`. Its `InferShape` method should be like following

```cpp
void TableLookupGrad::InferShape(context) {
  ...
  context.SetDataType("Embedding.Grad", kSelectedRows);
}
```


## Sparse Operators

There are several operators should be written to support `SelectedRows`. They are:

1. Operators which generates `SelectedRows` gradient. e.g. Gradient of `TableLookupOp`.
2. Optimize operators which support `SelectedRows` gradient. e.g. `SGD` or `AdaGrad` for `SelectedRows`. However, there should be only one `SGD` operator. `OpWithKernel::Run` should select a suitable kernel for both `dense` tensor or `SelectedRows`.
