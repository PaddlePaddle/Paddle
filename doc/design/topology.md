# Topology Overview
Topology is a concept in Paddle for representing neural networks.  A neural network contains one topology, which describes how layers connected to each other, and many parameters. The other deep learning frameworks may call this concept a computation graph, neural network configurations.

The topology is not only an API level concept but also how we organize the computation codes for each `Layer` or `Function` in Paddle. The Paddle should maintain a dictionary from `Layer Type` to Layer implementation, e.g.  from string `mul` to function `void tensor_multiply(Tensor& ins, Tensor& outs)'. The mechanism about how to manipulate topology by users, how Paddle maps user topology to implementations of `Layer` and `Function` is a fundamental problem for refactoring Paddle.

## User Stories and Examples

### Kernel Developers

Alan is a professional developer in CPU and GPU. He can write kernel functions of a new `Layer` with the best performance. However, he is not a familiar with Paddle API language, Python. Alan just needs to write the kernel function and register them in Paddle, and then Paddle should generate the user-side APIs for these kernel functions without any codes written by Alan.

```cpp
struct CosSimAttribute : public topology::Attribute {
  double scale;
  REGISTER_FUNC_ATTRIBUTE() {
    regAttr(&CosSimAttribute::scale, "scale", "the scale of cosine operator").defaultValue(1.0).largerThan(0.0);
  }
};

template <DeviceType devType>
void cos_kernel(std::vector<Tensor>& ins, std::vector<Tensor>& outs, const CosSimAttribute& attr) {
    // implemetation here.
}

BEGIN_REGISTER_FUNCTION(cosFwd, cosineForward, CosSimAttribute)
addTensor<INPUT>(/*dim*/ 2);
addTensor<INPUT>(/*dim*/ 2);
addTensor<OUTPUT>(/*shape = */ {topology::meta::kTensorShape_BATCH_SIZE, 1},
                  /*arg_type*/ ASSIGN_TO);

setDescription(R"DOC(Cosine similarity forward function.
There are two inputs of this function. The first matrix is a [h*w] matrix the
second input is a [h*w] matrix or a [1*w] matrix. the output matrix will be a
[h*1] matrix.
)DOC");

setShapeInferer([](std::vector<topology::TensorPtr>& ins,
                   std::vector<topology::TensorPtr>& outs) {
  auto& shape0 = ins[0]->shape();
  auto& shape1 = ins[1]->shape();

  if (shape0 != shape1 && (shape0[1] != shape1[1] || shape1[0] != 1))
    return Error(
        "Input shape should be same, or the second height should be 1");
  if (ins[0]->sequenceType() != ins[1]->sequenceType())
    return Error("Input sequence type should be same");
  outs[0]->setShape({ins[0]->shape()[0], 1});
  outs[0]->setSequenceType(ins[0]->sequenceType());
  outs[0]->setDataType(ins[0]->dataType());
  return Error();
});
END_REGISTER_FUNCTION(cosFwd);
```

### QA Developers

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

### Language Binding Developers

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
    ```cpp
    template <typename T>  // T is attribute type
    class Constraints {
    private:
      // (T* attr, bool alreadySet) -> Error
      // attr is an inout parameter for attribute.
      // alreadySet means if this attribute is set by user or previous callbacks.
      // return Error if the attribute is not valid.
      std::vector<std::function<Error(T*, bool)>> callbacks_;
    public:

      // Each constraint function will add a check function to callbacks_;
      Constraints<T>& mustSet();
      Constraints<T>& defaultValue(const T& val);
      Constraints<T>& largerThan(const T& val);

      // More constraint function below.
    };
    ```

* AttributeMeta:  It represent a meta information of an attribute, e.g. `scale`. It contains the attribute name,  description, type information and `Constraints`.
    ```cpp
    class AttributeMeta {
    public:
      std::string name;         // e.g. "scale"
      std::string description;  // e.g. "the scale of cosine operator"
      const std::type_info& type;  // e.g. type_id(double)

      std::any constraints;  // the contraints of this attribute. When implementation, 
                             // using `std::any` to get rid of template argument.
                             // But it is actually Constraints<T> actually, while T's type
                             // is store in the `type` field.
    };
    typedef std::shared_ptr<AttributeMeta> AttributeMetaPtr;
    ```
* AttributeMetaMap: The attribute meta map contains many AttributeMeta. Each Layer, FunctionMeta, TensorMeta is a AttributeMetaMap. User can addAttribute to an AttributeMetaMap.
    ```cpp
    class AttributeMetaMap: public std::unordered_map<std::string, AttributeMetaPtr> {
    public:

      /// Add attribute to map, returns constraints object, which user can add constraints.
      /// @code
      ///   attr_map.addAttribute("scale", "the scale of cosine operator").defaultValue(1.0).largerThan(0.0);
      /// @endcode
      template <typename T>
      Constraints<T>& addAttribute(const std::string& name, const std::string& description);
    };
    ```

* AttributeMap: The attribute map is the data structure which save attributes. The AttributeMap is not only used by user defined topology information, but also used by some meta information, which make meta information can store any type of Attributes, and be decoupled with upper invoker.
    ```cpp
    typedef std::unordered_map<std::string, std::any> AttributeMap;
    ```

* TensorMeta: Tensor is the input/output of the Layer or Function. It is an vector `AttributeMetaMap`. The data type, sequence type is just an attribute of the tensor.
    ```cpp
    enum DataType {DENSE, SPARSE, ...};
    enum SequenceType {NO_SEQUENCE, SEQUENCE, ...};
    class TensorMeta: public AttributeMetaMap {
    public:
      TensorMeta& setValidDataType(std::set<DataType> dataType);
      TensorMeta& setValidSeqType(std::set<SeqType> seqType);
      TensorMeta& setShapeDim(size_t dim);

    private:
      // nothing! TensorMeta just a AttributeMetaMap, but add some helper functions.
    };
    ```

* FunctionMeta: It represent a meta information of a paddle::Function. It contains two vectors of TensorMeta, and they are inputs and outputs. The FunctionMeta is a AttributeMetaMap. that kernel developers can add the attributes used by their kernel.
    ```cpp
    class FunctionMeta : public AttributeMetaMap {
    public:
      std::vector<TensorMetaPtr> inputs;
      std::vector<TensorMetaPtr> outputs;

      /// `cpuKernel` `gpuKernel` `shapeInferer`, `estimateFlops` just store into metaAttrs. 
      /// The invoker of function meta will decide which type and name should be.
      /// Also each field could be many types.
      /// For example,
      ///   shapeInferer is a function from `std::vector<Tensor>& in` to `std::vector<Tensor>& out`.
      ///   but different function could use different Attribute class.
      ///   cosine layer's shapeInferer could be
      ///   (std::vector<Tensor>& in, std::vector<Tensor>& out, const CosSimAttribute& attr) {...}
      AttributeMap metaAttrs;
    };
    ```

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
