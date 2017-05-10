# Topology Overview
Topology is a concept in Paddle for representing neural networks.  A neural network contains one topology, which describes how layers connected to each other, and many parameters. The other deep learning frameworks may call this concept a computation graph, neural network configurations.

The topology is not only an API level concept but also how we organize the computation codes for each `Layer` or `Function` in Paddle. The Paddle should maintain a dictionary from `Layer Type` to Layer implementation, e.g.  from string `mul` to function `void tensor_multiply(Tensor& ins, Tensor& outs)'. The mechanism about how to manipulate topology by users, how Paddle maps user topology to implementations of `Layer` and `Function` is a fundamental problem for refactoring Paddle.

## User Stories and examples

### Kernel Developers

Alan is a professional developer in CPU and GPU. He can write kernel functions of a new `Layer` with the best performance. However, he is not a familiar with Paddle API language, Python. Alan just needs to write the kernel function and register them in Paddle, and then Paddle should generate the user-side APIs for these kernel functions without any codes written by Alan.

```cpp
template <DeviceType devType>
void cos_kernel(std::vector<Tensor>& ins, std::vector<Tensor>& outs,  double scale) {
    // implemetation here.
}

BEGIN_REGISTER_FUNCTION(cos, cos_kernel)
// The parameter of cos function. 
func.addAttribute("scale", "The scale of cos layer").defaultValue(1.0).largerThan(0.0);

// Two inputs
func.addInput().dataType(Dense).dimension(2).supportSeqType();
func.addInput().dataType(Dense).dimension(2).supportSeqType();

// One outputs
func.addOutput().dataType(Dense).dimension(2).supportSeqType();

// Tell Paddle how to inference the output shape?
func.setShapeInferer([](std::vector<Dims>& ins, std::vector<Dims>& outs){
    outs[0] = {ins[0][0], 1};  // output dimension = batch_size * 1
});

END_REGISTER_FUNCTION()
```

### QA developer

Bob is a QA developer of Paddle.  He wants to tests all Paddle supported `Function` and `Layer`.  However, each layer has different configuration attributes, e.g. `scale` in `cosine` function. Each configuration attribute has different value range, data type. Bob should easily test all boundary conditions of one Layer or Functions by using new mechanism about topology.

```
auto cos = function::Register("cos");

for each_attribute in cos.attributes:
    each_attribute = each_attribute.min_value

test(cos);

for each_attribute in cos.attributes:
    each_attribute = each_attribute.max_value
test(cos);
```

### Language Binding developer

Carol is a language binding developer of Paddle. She wants to develop a language binding of Paddle. She is not familiar with Paddle C++ core and does not want to go so deep in Paddle. She just wants a clear list of what Layer Paddle supports, the configuration parameters of each Layer.

Also as a language binding developer, Carol does not want to write any topology validation code in language binding because Paddle C++ Core could be in flux and layer's API could be changed.

She just can access the register information of `Topology` and uses this information in another language. She can either uses reflection or code generation in that language to generate end-user APIs.

```python
import paddle

for layer_name in paddle.topology.meta.all_registed_layers:
    def __func__(**kwargs):
        layer_meta = paddle.topology.meta.all_registed_layers["layer_name"]
        return layer_meta.new_configration(kwargs)

    globals()[layer_name] = __func__
```

### API End-Users

David is a new user of Paddle, who are not familiar with Paddle and deep learning. He writes a Python program and configures a neural network. When he run this program, he expects a clear error message when his configuration is wrong. The error message should be like `cosine layer's scale parameter should be larger than 0.0.`, not just a `check error` in our computation kernel. Because we register all parameter's meta information, it is easy to achieve this goal.


## Goals

After thinking lots of user stories, we make the conclusion of what we want in Topology design.

* User should directly operate C++ topology configuration because we should maintain the consistency between each language bindings, and make language binding layer thin and easily to develop.
* Our topology configuration should be able to validate user's input and give a reasonable error message. Also, we should maintain some meta information of each configuration attribute, e.g. `scale` attribute in `cos` layer is a `double` value, should be larger than 0.0, and the default value is 1.0.
* We should serialize our topology into a portable format, so users can use the model they trained before for inference.
* We should let our kernel developer easily to register their kernel functions to Paddle and not make them write configuration APIs in Python.

## Implementation

### Meta Information
To achieve goals above, we need a place to store meta information of each layer. The meta information is used to describe what a layer could be configured, what the attributes of one layer could set, what the input types could be.

For example, the cosine layer should have two inputs, and the two inputs should be the same shape. The two inputs should both be the dense matrix. The cosine layer should have only one output, and the output shape should be [batch_size, 1] because, for each pair of input sample, the cosine similarity should be a scalar. The cosine layer has one configurable argument, `scale`. It is the scalar number multiplied to the cosine similarity.  `scale` should be a `double` value,  the default value is 1.0,  and should be larger than 0.0.

All these meta information should be written in namespace `paddle::topology::meta`. There are several basic classes in this namespace.

* Constraints:  It is a function list which stores the constraints of one attribute. It used to validate user input must be correct.
* AttributeMeta:  It represent a meta information of an attribute, e.g. `scale`. It contains the attribute name,  description, type information and `Constraints`.
* TensorMeta: Tensor is the input/output of the Layer or Function. It contains a vector of `AttributeMeta`. The data type, sequence type is just an attribute of the tensor.
* FunctionMeta: It represent a meta information of a paddle::Function. It contains two vectors of TensorMeta, and they are inputs and outputs. The FunctionMeta also contains a vector of AttributeMeta, that kernel developers can add the attributes used by their kernel.
* LayerMeta: A similar concept like FunctionMeta, but used to represent `Layer'.
* TopologyMeta: A topology meta contains a vector of `AttributeMeta`, which represent the attributes can be set globally in a topology.

### Topology information

The topology information is the actual information of a neural network. It is one to one correspondence to meta information. We use `std::any`(a.k.a `boost::any`) to represent the attribute value of each attribute because attribute could be any type(double/int/vector<int>, etc).

So the `topology::Tensor` contains an attribute map, e.g. `map<string, any>`.  The `Function` contains an attribute map, input tensors, and output tensors. The rest types of topology information are correspondent to its meta information.

## Step by step approach

After building the `Topology` concept in C++, Paddle's Python code could be clean up. However, the development process would be broken down into step by step, carefully completed, to make Paddle code steady and not introduce bugs.

The step by step approach are:

1. Add `Constraints`, `AttributeMeta` , `TensorMeta`, `FunctionMeta` to refactor the `paddle::Function` package. Make `paddle::Function` just a plain function registered to `FunctionMeta`. Use a small scope experiment make sure we could uses `topology::meta` and `topology` represent a piece of neural network.

2. Complete the `LayerMeta`, `TopologyMeta`, etc. But write a conversion method from `protobuf::LayerConfig`/`protobuf::ModelConfig` to `topology::Layer`/`topology::Topology`. Make `paddle_trainer` can use and test `topology` package. A side-effect of this job is to let `paddle_trainer` validation users' `trainer_config.conf` file, and give a reasonalbe error message when user gives a wrong configuration.

3. Clean up the implementation of `paddle.v2` topology. Let `v2` package not invoke `trainer_config_helper`, just invoke `topology` package directly from C-API.
