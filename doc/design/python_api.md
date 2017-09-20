# Design Doc: Python API

The top level user API in Python should be as same as API in `paddle.v2` after refactoring Paddle from a layer based framework to an operator based framework. There are many new classes in C++ in [compile time] for describing neural networks, such as `Variable`, `Operator`, `Block`. The issue about current design is how to give a proper way to wrap the C++ API to `paddle.v2` API and writing layers in Python.

This implementation of Python API includes two steps.

1. Implement the Python API using current C++ runtime concepts.
2. Replace the implementation by using compile-time concepts when they are completed.

The implementation of the first step is a temporary implementation. We should design our Python API concepts based on `compile-time` concepts. We just use `runtime` classes to implement it for now.


## Python Class and compile-time protobuf

Since we design our Python API concepts based on `compile-time`, we try to map our Python classes to every compile-time result, i.e., the protobuf messages. They are:


| Python Class | Compile-time protobuf |
| --- | --- |
| Block | BlockDesc |
| Operator | OpDesc |
| Variable | VarDesc |


### Block

Block is just like programming languages `{}`, which contains many operators and variables. There are two data fields in `Block`.  1) An associate map, whose key is variable name and value is variable itself; 2) A list of operators.

The block is hierarchical because PaddlePaddle supports RNN and IfElse. For example, RNN is like `for-loop` in programming languages. There is new `block` inside a `for-loop`. To represent hierarchies, `Block` stores the `parent Block` inside. If `parent=None`, the `Block` is the outermost block, i.e., the `global` block.


```python
class Block(objects):
    def __init__(self, parent=None):
        self.vars = map<string, Variable>()
        self.ops = vector<Operator>()
        self.parent = parent
    
    def create_var(self, ...):
        # create variable in `self.vars`
        return Variable(...)
    
    
    def create_global_var(self, ...):
        if self.parent is not None:
            return self.parent.create_global_var(...)
        else:
            return self.create_var(...)
    
    def create_parameter(self, ...):
        return self.create_global_var(...)
    
    def append_operator(self, ...):
        self.ops.append(...)
        
    def prepend_operator(self, ...):
       self.ops.prepend(...)
```

Users are able to create a global variable inside any block since they many create parameters inside a RNN or IfElseOp. All parameters should be stored in the global block, not the step block in RNN.

Users can create local variables for outputs of operators. Users can also append and prepend an operator in current block. Prepending `random initialize` operator or `load` operator is very useful to initialize parameters before training.


### Operator

Operator class will take inputs, outputs and attributes of the operator into `protobuf` OpDesc and create a C++ `OpDesc` instance. The `infer_shape` perform on C++ objects.

```python
class Operator(object):
    def __init__(self, type, inputs, outputs, attrs):
        # create OpDesc in Python
        op_desc = ...
        self.cpp_op_desc_ptr = core.OpDesc(op_desc)
        cpp.infer_shape(self.cpp_op_desc_ptr, inputs, outputs)

    def type(self):
        return self.cpp_op_desc_ptr.type()
```

After creating a C++ `OpDesc`, `Operator` in Python can only reads the attribute from C++ side.

### Variable

Operators' inputs, outputs, and parameters are all variables. In our design, a variable has four key attributes: its name(`name`), the block it belongs to(`block`), a pointer pointed to its C++ Protobuf object(`cpp_var_desc_ptr`), and the operator it is created by(`op`). All of these attributes are initialized in the constructor, except the `op`. The `op` will keep being `None` till the variable is taken as an operator's output.

```python
class Variable(object):
    def __init__(self, shape, dtype="float32", name=None, block=None):
        if name is None:
            name = unique_name_generator()
        self.name = name
        self.block = block
        # build C++ Protobuf object
        self.cpp_var_desc_ptr = ...
        self.op = None

    def shape(self):
        cpp_shape = self.cpp_var_desc_ptr.shape()
        return [None if elem < 0 else elem for elem in cpp_shape]
```

The Protobuf object should be created in C++ not Python because it is needed by infershape, and infershape is implemented by C++ code. The C++ Protobuf object is accessible for Python through the `cpp_var_desc_ptr`, just like how `shape()` function does.

The user is allowed to build a variable without specifying its name. If so, it is going to be assigned with an automatically generated unique name.

### Parameter

The parameter is a kind of special variable. They need to be initialized at the very beginning and updated after each batch training. So if a variable is a parameter, our compiler will add an initializer op and an optimizer op for it during the building process of computation graph. Apart from these, there is no more difference between variable and parameter. In other words, 'parameter' is only a label attached to variables, to tell the compiler these ones require additional processing.

```python
class Parameter(Variable):
    def __init__(self, trainable, initialize_attrs, optimize_attrs):
        pass
```

The class `Parameter` is derived from class `Variable`. In addition to variables have, parameters are able to hold their initializing and updating information. A parameter's `self.op` will always be `None` because it can never be an operator's output.


## Layer Functions

A layer is a Python function. When it is invoked, it creates a series of operators and variables then inserts them into the block. It is something like the macro in C++. It is called 'Layer' because the combination of added operators acts just like what a neural network layer does. 

Here are examples of how to write a data layer and FC layer:

### Data Layer

```python
def data_layer(name, type, block=None):
    if block is None:
        block = g_block
    # type = dense_vector(size=10) / integer_value(range=10)
    return block.create_global_var(
            name=name, 
            shape=[None] + type.dims(), 
            dtype=type.dtype)

``` 

Before building new variables, we need to specify which block to use. If we don't, the default one `g_block` will be used. In the above `data_layer` code, a variable is created and be inserted into the root block to make it global. This variable is going to be used as input data of the whole network.

### FC Layer

```python
def fc_layer(input, size, block=None, ...):
    if block is None:
        block = g_block
    w = block.create_parameter(...)
    b = block.create_parameter(...)
    out = stack.create_var()
    op = block.append_operator(Operator("FC", X=input, W=w, b=b, Out=out))
    out.op = op
    return out
```

In the `fc_layer` code, we create two parameters(`w` and `b`), one variable(`out`) and one operator(`FC operator`), then insert all of them into the specified block.
