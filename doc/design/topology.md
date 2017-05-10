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
