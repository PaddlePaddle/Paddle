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

<!-- TODO -->

```python
class Variable(object):
    def __init__(self, shape, dtype="float32", name=None, block=None):
        if name is None:
            if prefix is not None:
                name = unique_name_generator(prefix)
            else:
                name = unique_name_generator("unknown")
        self.name = name
        self.block = block
        self.cpp_var_desc_ptr = ...
        self.op = None

    def shape(self):
        cpp_shape = self.cpp_var_desc_ptr.shape()
        return [None if elem < 0 else elem for elem in cpp_shape]
```

### Parameter

<!-- 虽然Parameter不是编译器的概念，但是Python维护一个Parameter可以帮助我们构造计算图，知道哪个参数是可更新的等等 -->

<!-- 参数 is a special Variable -->

```pythonVclass Parameter(Variable):
    def __init__(self, trainable, initialize_attrs, optimize_attrs):
        pass
```

## Layer Functions

<!-- 给出一个Demo如何写Data Layer和FC Layer -->
