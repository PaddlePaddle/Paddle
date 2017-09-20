# Design Doc: ProgramDesc

The basic structure of a PaddlePaddle program is some nested blocks, as a C++ or Java program.

As described in [graph.md](./graph.md), the first five lines of the following PaddlePaddle program

```python
x = layer.data("images")
l = layer.data("label")
y = layer.fc(x)
cost = layer.mse(y, l)
optimize(cost)
train(cost, reader=mnist.train())
```

generates, or compiles, a PaddelPaddle program, which is represented by the following protobuf message:

```protobuf
message ProgramDesc {
  repeated BlockDesc blocks = 1;
}

message BlockDesc {
  required int32 parent = 1;
  repeated VarDesc vars = 2;
  repeated OpDesc ops = 3;
}

message OpDesc {
  AttrDesc attrs = 1;
  ...
}

message AttrDesc {
  required AttrType type = 1;

  // index into ProgramDesc::blocks when type==BLOCK
  optional int32 block = 2;
  ...
}
```

When each of the first five lines runs, related Python function, e.g., `layer.fc`, calls C++ InferShape functions.  This InferShape function needs to access the properties of VarDesc's accessed by the current OpDesc. These VarDesc's might not be defined in the current block, but in some ancestor blocks.  This requires that we can trace the parent of a block.

A nested block is often an attribute of an operator, most likely, an IfElseOp or a WhileOp.  In above solution, all blocks are in `ProgramDesc::blocks`, this implicitly assigns a zero-based ID to each block -- the index of the block in `ProgramDesc::blocks`.  So that `AttrDesc::block` could be an integer block ID.

With this design, the InferShape function should take the following parameters:

```c++
void InferShape(int current_block,
                int current_operator,
                ProgramDesc* program // might change VarDesc values.
                ) {
  ...
}
```

where

- `current_block` indices into `ProgramDesc::blocks`,
- `current_operator` indices into `BlockDesc::ops`.
