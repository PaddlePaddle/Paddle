# Design Doc: PaddlePaddle Programs

## Compile and Execution

A PaddlePaddle program consists of two parts -- the first generates a `ProgramDesc` protobuf message that describes the program, and the second runs this message using a C++ class `Executor`.

A simple example PaddlePaddle program can be found in [graph.md](../others/graph.md):

```python
x = layer.data("images")
l = layer.data("label")
y = layer.fc(x)
cost = layer.mse(y, l)
optimize(cost)
train(cost, reader=mnist.train())
```

The first five lines of the following PaddlePaddle program generates, or, compiles, the `ProgramDesc` message.  The last line runs it.

## Programs and Blocks

The basic structure of a PaddlePaddle program is some nested blocks, as a C++ or Java program.

- program: some nested blocks
- [block](./block.md):
  - some local variable definitions, and
  - a sequence of operators

The concept of block comes from usual programs.  For example, the following C++ program has three blocks:

```c++
int main() { // block 0
  int i = 0;
  if (i < 10) { // block 1
    for (int j = 0; j < 10; j++) { // block 2
    }
  }
  return 0;
}
```

The following PaddlePaddle program has three blocks:

```python
import paddle as pd  // block 0

x = minibatch([10, 20, 30]) # shape=[None, 1]
y = var(1) # shape=[1], value=1
z = minibatch([10, 20, 30]) # shape=[None, 1]
cond = larger_than(x, 15) # [false, true, true]

ie = pd.ifelse()
with ie.true_block():  // block 1
    d = pd.layer.add_scalar(x, y)
    ie.output(d, pd.layer.softmax(d))
with ie.false_block():  // block 2
    d = pd.layer.fc(z)
    ie.output(d, d+1)
o1, o2 = ie(cond)
```

## `BlockDesc` and `ProgramDesc`

All protobuf messages are defined in `framework.proto`.

`BlockDesc` is straight-forward -- it includes local variable definitions, `vars`, and a sequence of operators, `ops`.

```protobuf
message BlockDesc {
  required int32 parent = 1;
  repeated VarDesc vars = 2;
  repeated OpDesc ops = 3;
}
```

The parent ID indicates the parent block so that operators in a block can refer to variables defined locally and also those defined in their ancestor blocks.

All hierarchical blocks in a program are flattened and stored in an array. The block ID is the index of the block in this array.

```protobuf
message ProgramDesc {
  repeated BlockDesc blocks = 1;
}
```


### Global Block

The global block is the first one in the above array.

## Operators that Use Blocks

In the above example, the operator `IfElseOp` has two blocks -- the true branch and the false branch.

The definition of `OpDesc` shows that an operator could have some attributes:

```protobuf
message OpDesc {
  AttrDesc attrs = 1;
  ...
}
```

and an attribute could be of type block, which is, in fact, a block ID as described above:

```
message AttrDesc {
  required string name = 1;

  enum AttrType {
    INT = 1,
    STRING = 2,
    ...
    BLOCK = ...
  }
  required AttrType type = 2;

  optional int32 block = 10; // when type == BLOCK
  ...
}
```

## InferShape

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
