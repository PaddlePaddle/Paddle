# Design Doc: Python API

The top level user API in Python should be as same as API in `paddle.v2` after refactoring Paddle from a layer based framework to an operator based framework. There are many new classes in CPP in [compile time] for describing neural networks, such as `Variable`, `Operator`, `Block`. The issue about current design is how to give a proper way to wrap the C++ API to `paddle.v2` API and writing layers in Python.

This implementation of Python API includes two steps.

1. Implement the Python API using current C++ runtime concepts.
2. Replace the implementation by using compile-time concepts when they are completed.

The implementation of the first step is a temporary implementation. We should design our Python API concepts based on `compile-time` concepts. We just use `runtime` classes to implement it for now.


## Python Class and compile-time protobuf

As we design our Python API concepts based on `compile-time`, we try to map our Python classes to every compile-time result, i.e., the protobuf messages. They are:


| Python Class | Compile-time protobuf |
| --- | --- |
| Block | BlockDesc |
| Operator | OpDesc |
| Variable | VarDesc |


### Block

<!-- TODO -->

```python
class Block(objects):
	def __init__(self, parent=None):
		self.vars_ = map<string, Variable>()
		self.ops_ = vector<Operator>()
		if parent is None:
			self.global_vars = map<string, Variable>()
			self.parent=None
		else:
			self.parent = parent
			self.global_vars = None
	
	def create_global_vars(...):
		if self.parent is not None:
			return self.parent.create_global_vars(...)
		else:
			return self.global_vars.new()
```


### Operator

<!-- TODO -->

```python
class Operator(object):
	def __init__(self, type, inputs, outputs, attrs):
		# create OpDesc in Python
		op_desc = ...
		self.cpp_op_desc_ptr = cpp.to_cpp_op_desc(op_desc)
		cpp.infer_shapes(self.cpp_op_desc_ptr, inputs, outputs)
		outputs.op = self

	def type(self):
		return self.cpp_op_desc_ptr.type()
```

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

```python
class Parameter(Variable):
    def __init__(self, trainable, initialize_attrs, optimize_attrs):
        pass
```

## Layer Functions

<!-- 给出一个Demo如何写Data Layer和FC Layer -->
