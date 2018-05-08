# Design Doc: Selected Rows

`SelectedRows` is a type of sparse tensor data type, which is designed to support `embedding` operators. The gradient of embedding table is a sparse tensor. Only a few rows are non-zero values in this tensor. It is straight-forward to represent a sparse tensor by the following sparse tensor data structure:

```cpp
class SelectedRows {
 private:
  vector<int> rows_;
  Tensor value_;
  int height_;
};
```

The field `height_` is the first dimension of `SelectedRows`. The `rows` are the indices of the non-zero rows of `SelectedRows`. The `value_` field is an N-dim tensor of shape `[rows.size() /* NUM_ROWS */, ...]`, which supplies values for each row. The dimension of `SelectedRows` satisfies `[height_] + value_.shape[1:]`.

Suppose that a SelectedRows-typed variable `x` has many rows, but only two of them have values -- row 73 is `[1, 2]` and row 84 is `[3, 4]`, the `SelectedRows` representation would be:

```
x = SelectedRow {
  rows = [73, 84],
  value = [[1, 2], [3,4]]
}
```


## SelectedRows in Protobuf

`SelectedRows` is a type of `Variable`. `VarDesc` in protobuf should describe the `SelectedRows` information. Only the tensor dimension of a `SelectedRows` will be described in compile-time because the `rows_` and `value_` are dependent on the training data. 
So we use `TensorDesc` to unify `data_type` and `dims`. A LodTensorDesc contains a `TensorDesc` and `lod_level`. The description of `SelectedRows` is a Tensor description.

```proto
message TensorDesc {
  required DataType data_type = 1;
  repeated int64 dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
}

message LodTensorDesc {
  required TensorDesc tensor = 1;
  optional int lod_level = 2;
}

message VarDesc {
  required string name = 1;
  enum VarType { 
    LOD_TENSOR = 0;
    SELECTED_ROWS = 1;
  }
  required VarType type = 2;
  optional LodTensorDesc lod_desc = 3;
  optional TensorDesc selected_rows_desc = 4;
  optional bool persistable = 5 [ default = false ];
}
```

## InferShape for Selected Rows

Just like `LoD` information, `InferShape` method will infer the output tensor type as well. The operator should decide whether its output is a `SelectedRows` or `Dense` tensor.

For example, the gradient operator of `TableLookup` will always generate `SelectedRows`. Its `InferShape` method should be like following

```cpp
void TableLookupGrad::InferShape(context) {
  ...
  context.SetDataType("Embedding.Grad", kSelectedRows);
}
```


## Sparse Operators

There are several operators that need to be written to support `SelectedRows`. These are:

1. Operators which generate `SelectedRows` gradient. e.g. Gradient of `TableLookupOp`.
2. Optimize operators which support `SelectedRows` gradient. e.g. `SGD` or `AdaGrad` for `SelectedRows`. However, there should be only one `SGD` operator. `OpWithKernel::Run` should select a suitable kernel for both `dense` tensor or `SelectedRows`.
