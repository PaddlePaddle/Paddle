# Design Doc: Python API

Due to the refactorization of the PaddlePaddle core, we need Python classes to construct corresponding protobuf messages that describe a DL program.

<table>
<thead>
<tr>
<th>Python classes</th>
<th>Protobuf messages</th>
</tr>
</thead>
<tbody>
<tr>
<td>Program </td>
<td>ProgramDesc </td>
</tr>
<tr>
<td>Block  </td>
<td>BlockDesc </td>
</tr>
<tr>
<td>Operator </td>
<td>OpDesc </td>
</tr>
<tr>
<td>Variable </td>
<td>VarDesc </td>
</tr>
</tbody>
</table>


Please be aware that these Python classes need to maintain some construction-time information, which are not part of the protobuf messages.

## Core Concepts

### Program

A `ProgramDesc` describes a [DL program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md), which is composed of an array of `BlockDesc`s.  The `BlockDesc`s in a `ProgramDesc` can have a tree-like hierarchical structure. However, the `ProgramDesc` onlys stores a flattened array of `BlockDesc`s. A `BlockDesc` refers to its parent block by its index in the array.  For example, operators in the step block of an RNN operator need to be able to access variables in its ancestor blocks.

Whenever we create a block, we need to set its parent block to the current block, hence the Python class `Program` needs to maintain a data member `current_block`.

```python
class Program(objects):
    def __init__(self):
        self.desc = core.NewProgram() # a C++ ProgramDesc pointer.
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

A [Block](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/block.md) includes

1. a map from variable names to an instance of the Python `Variable` class, and
1. a list of `Operator` instances.

```python
class Block(objects):
    def __init__(self, program, parent_idx):
        self.desc = core.NewBlock(program.desc)
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
        self.desc = core.NewOpDesc(block.desc, type, inputs, outputs, attrs)
        core.infer_shape(self.desc, inputs, outputs)

    def type(self):
        return self.desc.type()
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
        self.desc = core.NewVarDesc(block.desc, name, shape, lod_level)
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

## Layer Function

A layer is a Python function that creates some operators and variables. Layers simplify the work of application programmers.

Layer functions take `Variable` and configuration parameters as its input and return the output variable(s).

For example, `FullyConnected` take one or more variable as its input. The input could be input data or another layer's output. There are many configuration options for a `FullyConnected` layer, such as layer size, activation, parameter names, initialization strategies of parameters, and so on. The `FullyConnected` layer will return an output variable.


### Necessity for reusing code between layer functions

There are a lot of code that can be reused. Such as

* Give the default value of configuration. e.g., default initialize strategy for parameters is uniform random with `min = -1.0`, `max = 1.0`. and default initialize strategy for bias is to fill zero.
* Append the activation operator.
* Create a temporary variable.
* Create parameter.
* Generate a unique name.
* Add a bias.
* ...

A mechanism to reuse code between layer functions is necessary. It will be around [150 lines of code](https://github.com/PaddlePaddle/Paddle/pull/4724/files#diff-823b27e07e93914ada859232ae23f846R12) if we write a `FullyConnected` layer without any helper functions.



### Comparision between global functions and helper class

The `FullyConnected` layer will be as follow when we provide global functions:

```python
def fc_layer(input, size, param_attr=None, bias_attr=None, act=None, name=None):
  if name is None:
    name = unique_name("fc")
  input = multiple_input(input)
  param_attr = default_param_attr(param_attr)
  param_attr = multiple_param_attr(param_attr, len(input))

  # mul
  mul_results = []
  for ipt, attr in zip(input, param_attr):
    shape = ipt.shape[1:] + [size]
    w = g_program.global_block().create_parameter(shape, ipt.dtype, name, attr)
    tmp = create_tmp_var(name)
    g_program.current_block().append_op("mul", {ipt, w}, {tmp})
  mul_results.append(tmp)

  # add sum
  ...
  # add bias
  ...
  # add activation
  ...
  return out
```

We can provide many helpers functions for layer developers. However, there are several disadvantages for global helper functions:

1. We need a namespace for these methods, then layer developers can quickly figure out what method they can use.
2. Global functions will force layer developers to pass its parameter time by time.

So we provide a helper class, `LayerHelper`, to share code between layer functions. The `FullyConnected` Layer will be as follow.

```python
def fc_layer(input, size, param_attr=None, bias_attr=None, act=None, name=None):
  helper = LayerHelper(locals())  # pass all parameter to LayerHelper

  mul_results = []
  for ipt, param in helper.iter_multiple_input_and_param():
    w = helper.create_parameter(shape=ipt.shape[1:] + [size], dtype = ipt.dtype)
    tmp = helper.create_tmp_variable()
    helper.append_op('mul', {ipt, w}, {tmp})
    mul_results.append(tmp)

  pre_bias = helper.add_sum(mul_results)
  pre_activation = helper.add_bias(pre_bias)
  return helper.add_activation(pre_activation)
```

We not only use the fewer lines of code to write `fc_layer` but also make the code clearer to understand. At the same time, layer developers can figure out what function they can invoke by typing `helper.` in a python editor.


### Implementation of layer helper

We just keep all parameters of a layer function as a dictionary in layer helper as a private data member. Every method of layer helper will look up the dictionary after it is invoked. In that way, we can implement a layer helper for all layer functions even some layer does not contain some operator. For example, The `activation` is used by the FullyConnected layer or convolution layers, but a cross-entropy layer does not use it. The example code of `add_activation` are:

```python
class LayerHelper(object):
  def __init__(self, **kwargs):  # kwargs is short for `keyword arguments`
    self.kwargs = kwargs

  def add_activation(self, input_var):
    act = self.kwargs.get("act", None)  # default value is None
    if act is None:  # do nothing if no act
      return input_var

    tmp = self.create_tmp_var(self)
    self.append_op(type=act, input=input_var, output=tmp)
    return tmp
```

### Return value of layer functions

The layer will return a Variable, which is also the output of an operator.  However, outputs of a layer function have more attributes than an operator. There are parameter variables, and their gradient variables need to return. To return them is useful. For example,

1. Users can debug the network by printing parameter gradients.
2. Users can append attributes to a parameter, such as, `param.stop_gradient=True` will make a parameter stop generate the gradient. We can fix the parameter value during training by using this attribute.

However, it is good to return a Variable for layers, since all layers and operators use Variables as their parameters. We can just append a `param` field and a `grad` field for layer function since the Python is dynamic typing.

The sample usage is

```python
data = fluid.layers.data(...)
hidden = fluid.layers.fc(data, ...)
...

executor.run(fetch_list=[hidden.param, hidden.param.grad], ...)
```


## Optimizer

[Optimizer Design Doc](./optimizer.md)
