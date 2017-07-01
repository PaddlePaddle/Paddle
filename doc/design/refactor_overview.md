# Overlook of each level design

## User Level API

* Scope
	* High level API, global has a default `Scope`
* Variable
	* User can create_var. model's layer function always return variable.
	* Variable uses name to know category.
		* xxx means the input or output of layer
		* xxx_param means the parameter of layer
		* xxx_grad means the gradient of input or output.
		* xxx_param\_grad means the parameter's gradient of layer.
* Model
	* High level API, i.e., Layer API.
	* Just like current v2 API.
	* It maintains:
		* A Scope.
		* Utility networks, like init-param, save, load.
		* A network topology.
	* model.xx_layer's input can be a string or variable. If it is string, then that name should be used to create variable by model.scope

```python
import paddle

model = paddle.framework.model()
hidden = model.fc_layer(input="img", size=200, bias=True, activation="sigmoid")  # `hidden` is a variable.
hidden = model.fc_layer(input=hidden, size=200, bias=True, activation="sigmoid")

# If user want to configure op, then you can use
# model.network.add_op(...) API, it is a low level python api.

prob = model.fc_layer(input=hidden, size=10, bias=True, activation="softmax")
loss = model.cross_entropy(input=probo, label="label")
avg_loss = model.mean(input=loss)
model.backward(from=avg_loss)
model.sgd(learning_rate=1e-4)

# Initialize all parameter variable, randomize
model.initialize_parameters()
# Or model.load_parameter(from="./model_path")

for mini_batch in paddle.dataset.mnist.trainset():
  img, lbl = mini_batch
  model.fill("img", img)
  model.fill("label", lbl)
  model.run()  # run a mini-batch
```

## Python Low-Level API

### Scope

```python
# filename: paddle/framework/scope.py

class Scope(paddle.cpp.framework.Scope):
  def create_variable(self, name):
    # call C++ code.
  def get_variable(self, name):
    # call C++ code.
  
  @staticmethod
  def new_scope(parent=None):
    # call C++ code.
    return Scope(...)

gScope = Scope.new_scope()
```

### Ops

```python
# filename: paddle/framework/ops.py
# ops.py is generated when compile.

def fc_op(input, w, b, output):
  # input, w, b, output are Variable
  desc = OpDesc()
  desc.inputs = [x.name for x in [input, w, b]]
  desc.outputs = [x.name for x in [output]]
  desc.type = "fc"
  return paddle.cpp.framework.NewOp(desc)

def sigmoid(input, output):
  # input, output are variable
  desc = OpDesc()
  desc.inputs = [x.name for x in [input]]
  desc.outputs = [x.name for x in [output]]
  desc.type = "sigmoid"
  return paddle.cpp.framework.NewOp(desc)
  
def softmax(input, output):
  ...  # similar implementation as above

def mean(input, output):
  ...  # similar implementation as above

def cross_entropy(input, label, output):
  desc = OpDesc()
  desc.input = [x.name for x in [input, label]]
  desc.output = [x.name for x in [output]]
  desc.type = "cross_entropy"
  return paddle.cpp.framework.NewOp(desc)
```

### Network

```python
# filename: paddle/framework/network.py
# Network it is just a wrapper for C++ NetworkBase
class Network(paddle.cpp.NetworkBase):
  def __init__(self):
    # create an empty network by default

  def add_op(self, op):
    # add op to network, op should be a C++ OperatorBase pointer.
    # add op will return a OpIndex. OpIndex is used by `run` to
    # execute partial of network.
    return OpIndex() 
  
  def run(self, scope, from=OpIndex(-1), to=OpIndex(-1)):
    # run that network, and modify scope.
    # from & to uses two OpIndex which is returned by `add_op`
    # Whole network will be executed by default, but user can
    # select part of network to execute.
    # Call C++ code.
```

### NetworkFunctions
```python
# filename: paddle/framework/network_funcs.py
# Network Functions are some function will modify a network, like `Backward`, `Optimize`, etc. 
# It should be implemented in C++, but expose API in Python.

def backward(network, from):
  # Call C++ code `void backward(NetworkBase* network, OpIndex from);`
  # It will appending all backward ops to `network.ops_` and return
  # the final op index, and gradient names.
  return OpIndex(), gradient_names

def optimize(network, opt_type, opt_attr, param_grad):
  assert isinstance(opt_type, str)
  assert isinstance(opt_attr, AttributeMap)
  assert isinstance(param_grad, map)
  # param_grad is map<string /* param's var name*/, string /* grad's var name*/> 
  # Call C++ code `void optimize(NetworkBase* network, std::string opt_type, AttributeMap attrs);`
  # It will `insert` or `append` all optimize ops to `network.ops_`.
  # Each pair of parameter and gradient, should create one optimize op
  # Return the network's final op index.
  return OpIndex()
```



### Model

```python
# filename: paddle/framework/model.py
# Model is a pure python object. 
# It take all python low level APIs, and provide the high level API
# like `paddle.v2`

import paddle.framework.ops
import paddle.framework.network_funcs

class Model(object):  # Pure python object
  def __init__(scope=None):
    if scope is None:
      scope = gScope  # default use global scope.
    self.scope = scope
    # 4 networks.
    # utility networks
   	self.randmon_param_network = Network()
   	
   	# real network
  	self.network = Network()
  	
  	self.param_grad = dict()  # param name to gradient name map
  	self.all_params = list()  # all parameter names
 
 def fc_layer(input, size, bias, activation, name=None):
   # prepare arguments
   if name is None:
     name = generate_unique_name(prefix="fc")
   if isinstance(input, str):
     input = self.scope.create_var(input)  # create variable
     
   # new/get variables
   w = self.scope.create_var(name + "_w_param")
   self.all_params.append(name + "_w_param")
   b = None
   if bias:
     b = self.scope.create_var(name + "_b_param")
     self.all_params.append(name + "_b_param")
   output = self.scope.create_var(name + "_fc_out")
   act_output = self.scope.create_var(name + "_out")
   
   # add ops to network
   self.network.add_op(paddle.framework.ops.fc_op(input, w, b, output))
   if activation == "softmax":
     self.network.add_op(paddle.framework.ops.softmax(output, act_output))
   elif activation == "sigmoid":
     self.network.add_op(paddle.framework.ops.sigmoid(output, act_output))
   
   # add ops to self.randmon_param_network
   # Pseudo code, could have some attribute for random op,
   # e.g., uniform random min and max value
   self.randmon_param_network.add_op(paddle.framework.ops.random(w))
   if b is not None:
     self.randmon_param_network.add_op(paddle.framework.ops.random(b))

   return act_output

  def mean(...): ... # similar implementation
  
  def cross_entropy(...): ... # similar implementation
  
  def backward(from=varname):
    # backward is from a op, not a variable.
    # But it is easier for user backward from a variable name.
    opidx = self.network.find_op_by_output(varname)
    _, grads = network_funcs.backward(self.network, opidx)
    self.param_grads = make_association_between_param_and_grads(self.all_params, grads)
    
  def optimize(...):
    network_funs.optimize(self.network, self.param_grads, ....)
    
  def run():
    self.network.run(self.scope)
    
  def initialize_parameters():
    self.randmon_param_network.run(self.scope)
  
  def fill(self, var_name, data):
    self.scope[var_name].fill(data)
```

## Python to C++ interface

### Python invoke C++ function

* Wrap C++ function into C-API.
	* paddle/framwork/c_api.h
* Python Directly invoke C-API by ctypes, cffi, or Cython.
	* ctypes is `dlopen`, `dlsym` API for Python, it make Python directly invoke C functions.
	* cffi is a wraper for `ctypes`. It read `c_api.h` and generate Python functions in runtime.
	* Cython is a 3rd-package for Python. It is a extension for Python language, make python can directly import dynamic library.
	* Either of them are fine.


### Generate Python `ops.py`

That ops.py are generated by C++ while compiling. It is user-friendly to have a `plain` ops.py instead of generating python method in runtime.

C++ side will create a binary named `ops_py_gen`. It takes all op's `OpProto`, which is a protobuf message for 3rd-party code generating, and generate `ops.py`.

The `OpProto` is shown below.

```proto
message VarProto {
  required string name = 1;
  required string comment = 2;
};
enum AttrType {
  INT = 1,
  FLOAT = 2,
  STRING = 3,
  INTS = 4,
  FLOATS = 5,
  STRINGS = 6
}
message AttrProto {
  required string name = 1;
  required string comment = 2;
  required AttrType type = 3;
};

message OpProto {
  repeated VarProto inputs = 1;
  repeated VarProto outputs = 2;
  required string comment = 3;
  repeated AttrProto attrs = 4;
  required string type = 5;
};
```

Each type of Op has one `OpProto`. For example, `FcOp`'s OpProto is

```text
{
  "inputs": [
    {"name":"input", "comment": "the input data of fc op."},
    {"name":"w", "comment": "the weight of fc op."},
    {"name":"b", "comment": "the bias of fc op."}
  ],
  "outputs": [
  	{"name", "output", "comment": "the output data of fc op."}
  ],
  "comment": "FC Op. output = input*w+b",
  "attrs": [],
  "type": "fc"
}
```

In generated `ops.py`, there are many op functions. Each op function will construct a `OpDesc` proto message. The op function use that `OpDesc` to invoke C++ API `CreateOp(const OpDesc&)`, and a op pointer will be created and returned.

That `OpDesc` is shown below

```proto
syntax="proto3";
message AttrDesc {
  AttrType type = 1;
  optional int i = 2;
  optional float f = 3;
  optional string s = 4;
  repeated int ints = 5;
  repeated float floats = 6;
  repeated string strings = 7;
};

message OpDesc {
  repeated string inputs = 1;
  repeated string outputs = 2;
  string type = 3;
  map<string, AttrDesc> attrs = 4;
};
```

The `OpDesc` uses `AttrDesc` as Op's attribute. The `AttrDesc` and `AttrProto` shared a same type `AttrType`. One of `i`, `f`, `s`, `ints`, `floats`, `strings`, which is associate with type, should be set.

The `OpDesc` contains the `inputs`, `outputs`, `type` and `attrs`. `inputs` and `outputs` are variable names from `Scope`. The `type` is registry name of Op.

## C++ interface Implementation

After we finish discuess `C++ <--> Python`, the C++ implementation is shown below.

### Network

The network is a configurable and runnable interface for User. The network interface shown below.

```cpp
using OpIndex = size_t;
class NetworkBase {
public:
  virtual OpIndex AddOp(std::unique_ptr<OpBase> op) = 0;
  virtual void Run(Scope* scope, OpIndex from = -1UL, OpIndex to=-1UL) = 0;
};
```

Because the Network is directly exposed to end user. So the `Run` method only takes `Scope` as parameter. It just modify variables in `Scope`. The `AddOp` is exposed to end user, too. So in `Python` user can directly manipulate `Network`.

### OpBase

```cpp
struct OpRunContext {
  vector<Variable* > inputs_;
  vector<Variable* > outputs_;
  DeviceContext* ctx_;
};

struct OpInferShapeContext {
  vector<DDim > inputDims_;
  vector<DDim > outputDims_;
};

class OpBase {
public:
  virtual void Run(OpRunContext* ctx) const = 0;
  virtual void InferShape(OpInferShapeContext* ctx) const = 0;
  vector<string> inputs_;
  vector<string> outputs_;
};
```

The `OpBase` is created by Python, but it is used by `NetworkBase`. 

* The creation of OpBase uses `OpDesc`. The `OpDesc`'s `inputs` and `outputs` will fill to `OpBase`'s constructor.
* The `Run()` method take an `OpRunContext*` as argument. That `OpRunContext*` is created by `NetworkBase` in `NetworkBase::Run()` method.
* The `InferShape` method take an `OpInferShapeContext*` as argument. Taht `OpInferShapeContext*` is also created by `NetworkBase` in `NetworkBase::Run()`. It will be invoked each time in `NetworkBase::Run`, because each mini-batch's data size could be changed.


### OpWithAttr

The `OpBase` is not handle each op's `AttrDesc`. Because each op's `Attr` type are not same. It should be Op's a template type. So the `OpBase` also erase the type for `Attr`.

There are a middle type to parse Op's attribute. It shown below

```cpp
template <typename T>
class OpWithAttr : public OpBase {
public:
  using ATTR_TYPE = T;
  T attrs_;
};
```

When register `Op`, the attribute type will register together. The sample register code like:

```cpp
struct CosineAttr {
  float scale_;
};

class CosineOp : public OpWithAttr<ConsineAttr> {
  ...
};

class CosineOpRegisterEntry : OpRegisterEntry<ConsineOp> {
public:
  CosineOpRegisterEntry() {
    AddInput("a", "the first input of cosine op");
    AddInput("b", "the second input of cosine op");
    AddOutput("output", "the output of cosine op");
    SetComment("The cosine op. output = cos(a, b)");
    AddAttr(&CosineAttr::scale_, "scale", "the scale of cosine op")
      .Default(1.0)
      .LargerThan(0.0);
  }
};

REGISTER_OP(cos, CosineOpRegisterEntry);
```

### OpWithKernel

Since we expose `Op` directly to end-user. We must make one `Op` can run on multiple devices, or switch devices in `CreateOp` method.

If we want to switch devices in `CreateOp` method, `CreateOp` must take `(Device, DataType)` as argument. But we cannot know what device and data type that Op should Run with, because the `Op` is used by `Network`. The `Network` can run on many device. Which device should run is not decided by `Op`.

Moreover, the kinds of device and data type is changed time by time. By handle writing a map from one `OpDesc` to multiple devices `Op` instance, that part of code will be flux.

The Operator in MXNet and Tensorflow contains many kernels. The key question I think is whether Kernel is needed or not. The key question is how to make `Op` simplest and logic clearest. If Kernel is a necessary concept for `Op`, it is clear that each `Op` contains many `kernel`. Then the question is how to make `Kernel` simple.

The simplest `Kernel` is a `std::function<void(OpRunContext*)>`. The `Kernel` should store in `Op` in an association map. The implementation of `OpWithKernel` likes below.

```cpp
struct KernelKey {
  Place place_;
  DataType type_;
  
  KernelKey(OpRunContext* ctx) {
    // KernelKey could create by OpRunContext.
  }
};


template <typename T>
class OpWithKernel: public OpWithAttr<T> {
public:
  // The attribute should be passed to OpWithKernel, too.
  using KernelType = std::function(void<OpRunContext*, const T&>);
  map<KernelKey, KernelType> kernels_;
  
  void Run(OpRunContext* ctx) const final {
    kernels_[KernelKey(ctx)](ctx, param_);
  }
};
```

Another related question is why divided two class `OpWithKernel` and `OpWithAttr`. Because only computation Op has `Kernel`, but all Op has `Attr`.

### The demo code for how to implement an Op.

We uses `CosineOp` as demo, the code shows below.

In `cosine_op.h`

```cpp
class CosineOpAttr {
  float scale_;
};

template <typename PlaceType, typename T>
void CosineOpKernel(OpRunContext* ctx, const CosineOpAttr& attr) {
  auto a = ctx->inputs[0].Get<PlaceType, T>();
  auto b = ctx->inputs[1].Get<PlaceType, T>();
  auto out = ctx->output[0].Get<PlaceType, T>();
  out.cosine_similarity(a, b);  // pseudo code here.
}

class CosineOp : public OpWithKernel<CosineOpAttr> {
public:
  void InferShape(OpInferShapeContext* ctx) const {
    auto& shape_a = ctx->inputShape[0];
    auto& shape_b = ctx->inputShape[1];
    PADDLE_ENFORCE(shape_a.size() == 2);
    PADDLE_ENFORCE(shape_b.size() == 2);
    PADDLE_ENFORCE(shape_a == shape_b);
    auto& oshape = ctx->outputShape[0];
    oshape = {shape_a[0], 1};
  }
};
```

In `cosine_op.cpp`, we register `ConsineOp` and CPU kernels.

```cpp
class ConsineOpRegisterEntry : public OpRegisterEntry<CosineOp> {
public:
  CosineOpRegisterEntry() {
    AddInput("a", "the first input of cosine op");
    AddInput("b", "the second input of cosine op");
    AddOutput("output", "the output of cosine op");
    SetComment("The cosine op. output = cos(a, b)");
    AddAttr(&CosineAttr::scale_, "scale", "the scale of cosine op")
      .Default(1.0)
      .LargerThan(0.0);
  }
};

REGISTER_OP(cos, ConsineOpRegisterEntry);
REGISTER_OP_KERNEL(cos, CPU, float, CosineOpKernel<CPU, float>);
```

In `cosine_op.cu`, we register GPU kernels.

```cpp
REGISTER_OP_KERNEL(cos, GPU, float, CosineOpKernel<GPU, float>);
```
