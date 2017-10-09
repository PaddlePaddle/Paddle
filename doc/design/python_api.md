# Design Doc: Python API

Due to the refactorization of the PaddlePaddle core, we need Python classes to construct corresponding protobuf messages that describe a DL program.

| Python classes | Protobuf messages |
| --- | --- |
| Program | ProgramDesc |
| Block | BlockDesc |
| Operator | OpDesc |
| Variable | VarDesc |

Please be aware that these Python classes need to maintain some construction-time information, which are not part of the protobuf messages.

## Core Concepts

### Program

A `ProgramDesc` describes a [DL program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/program.md), which is composed of an array of `BlockDesc`s.  The `BlockDesc`s in a `ProgramDesc` can have a tree-like hierarchical structure. However, the `ProgramDesc` onlys stores a flattened array of `BlockDesc`s. A `BlockDesc` refers to its parent block by its index in the array.  For example, operators in the step block of an RNN operator need to be able to access variables in its ancestor blocks.

Whenever we create a block, we need to set its parent block to the current block, hence the Python class `Program` needs to maintain a data member `current_block`.

```python
class Program(objects):
    def __init__(self):
        self.proto = core.NewProgram() # a C++ ProgramDesc pointer.
        self.blocks = vector<Block>()
        self.blocks.append(Block(self, -1)) # the global block
        self.current_block = 0          # initialized to the global block

    def global_block():
        return self.blocks[0]

    def current_block():
        return self.get_block(self.current_block)

    def rollback():
        self.current_block = self.current_block().parent_idx

    def create_block():
        new_block_idx = len(self.block)
        self.blocks.append(Block(self, self.current_block))
        self.current_block = new_block_idx
        return current_block()
```

`Program` is an accessor to the protobuf message `ProgramDesc`, which is created in C++ space, because the InferShape function is in C++, which manipulates `VarDesc` messages, which are in turn members of `BlockDesc`, which is a member of `ProgramDesc`.

`Program` creates the first block as the global block in its constructor.  All parameters and their initializer operators are in the global block.

### Block

A [Block](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/block.md) includes

1. a map from variable names to an instance of the Python `Variable` class, and
1. a list of `Operator` instances.

```python
class Block(objects):
    def __init__(self, program, parent_idx):
        self.proto = core.NewBlock(program.proto)
        self.program = program
        self.vars = map<string, Variable>()
        self.ops = vector<Operator>()
        self.parent_idx = parent_idx

    def create_var(self, ...):
        return Variable(self, ...)

    def _create_global_var(self, ...):
        program.global_block().create_var(...)

    def create_parameter(self, name, ...):
        # Parameter is a subclass of variable. See Parameter section for details.
        self.vars[name] = Parameter(self._create_global_var(...), ...)
        return self.vars[name]

    def append_operator(self, ...):
        self.ops.append(Operator(self, ...))

    def prepend_operator(self, ...): # Parameter's ctor prepands initialize operators.
       self.ops.prepend(Operator(self, ...))
```

`create_parameter` is necessary because parameters are global variables, defined in the global block, but can be created in some sub-blocks. For example, an FC layer in the step block of an RNN operator.

`prepend_operator` is necessary because the constructor of `Parameter` needs to create the initialize (or load) operator of the parameter, and would like to put it in the *preamble* of the global block.

### Operator

The `Operator` class fills in the `OpDesc` message and calls the C++ function `InferShape` to infer the output shapes from the input shapes.

```python
class Operator(object):
    def __init__(self,
                 block,  # Block
                 type,   # string
                 inputs, # dict<string, Variable>
                 outputs,# dict<stirng, Variable>
                 attrs   # dict<string, Any>
                 ):
        self.proto = core.NewOpDesc(block.proto, type, inputs, outputs, attrs)
        core.infer_shape(self.proto, inputs, outputs)

    def type(self):
        return self.proto.type()
```

`Operator` creates the `OpDesc` message in C++ space, so that it can call the `InferShape` function, which is in C++.

### Variable

Operators take Variables as its inputs and outputs.

```python
class Variable(object):
    def __init__(self,
                 block=None,      # Block
                 name=None,       # string
                 shape,           # tuple
                 dtype="float32", # string
                 lod_level=None   # int
                 ):
        if name is None:
            name = unique_name_generator()
        self.name = name
        self.block = block
        self.proto = core.NewVarDesc(block.proto, name, shape, lod_level)
        self.writer = None
```

Please be aware of `self.writer`, that tracks operator who creates the variable.  It possible that there are more than one operators who write a variable, but in Python space, each write to a variable is represented by a Variable class.  This is guaranteed by the fact that **`core.NewVarDesc` must NOT create a new `VarDesc` message if its name already exists in the specified block**.

### Parameter

A parameter is a global variable with an initializer (or load) operator.

```python
class Parameter(Variable):
    def __init__(self,
                 block=None,      # Block
                 name=None,       # string
                 shape,           # tuple
                 dtype="float32", # string
                 lod_level=None   # int
                 trainable,       # bool
                 initialize_op_attrs,
                 optimize_op_attrs):
        super(Parameter, self).__init__(block, name, shape, dtype, lod_level)
        self.trainable = trainable
        self.optimize_op_attrs = optimize_op_attrs
        block.prepend(Operator(block,  # Block
                               initialize_op_attrs['type'],   # string
                               None,   # no inputs
                               self,   # output is the parameter
                               initialize_op_attrs)
```

When users create a parameter, they can call

```python
program.create_parameter(
  ...,
  init_attr={
    type: "uniform_random",
    min: -1.0,
    max: 1.0,
  })
)
```

In above example, `init_attr.type` names an initialize operator.  It can also name the load operator

```python
init_attr={
 type: "load",
 filename: "something.numpy",
}
```

`optimize_op_attrs` is not in the `VarDesc` message, but kept in the Python instance, as it will be used in the Python space when creating the optimize operator's `OpDesc`, and will be in the `OpDesc` message.

## Layer Functions

A layer is a Python function that creates some operators and variables.  Layers simplify the work of application programmers.

### Data Layer

```python
def data_layer(name, type, column_name):
    block = the_current_program.glolal_block()
    var = block.create_global_var(
            name=name,
            shape=[None] + type.dims(),
            dtype=type.dtype)
    block.prepend_operator(block,
                           type="Feed",
                           inputs = None,
                           outputs = [var],
                           {column_name: column_name})
    return var
```

The input to the feed operator is a special variable in the global scope, which is the output of [Python readers](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/reader/README.md).

### FC Layer

```python
def fc_layer(input, size, ...):
    block = program.current_block()
    w = block.create_parameter(...)
    b = block.create_parameter(...)
    out = block.create_var()
    op = block.append_operator("FC", X=input, W=w, b=b, out=out)
    out.writer = op
    return out
```
