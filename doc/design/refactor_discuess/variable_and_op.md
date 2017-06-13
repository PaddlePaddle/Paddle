## Tensor, TensorAttr, TensorBuffer
```cpp
enum TensorType {kFLOAT32, kINT32};

struct TensorAttr {
  std::string name_;
  TensorType type_;
  bool needGrad_ {false};
  SmallVector<size_t, 4> dims_;
  int device{-1};
};

struct Tensor {
  TensorAttrPtr attr_;
  TensorBufferPtr buffer_;
};
```

TensorAttr是记录在计算图中，对某一个Op输入输出参数的描述。包括输入输出的名字，数据类型，形状，设备等等。

* name_ 在一个计算图中，每一个Tensor都有唯一的名字。
* type_ Paddle的Tensor类型按照数据类型分类，只包括float, double, int等类型。
* needGrad_ 这个Tensor是否需要在backward的时候计算梯度。
* dims_ 这个Tensor的维度信息。使用SmallVector是为了避免alloc小内存带来的开销。
* device_ 设备信息，默认是-1，即CPU。0即GPU-0。

TensorBuffer和Tensor并没有记录在计算图中，因为内存/显存具体值是多少，不是计算图描述中应有的属性。`Tensor`会被Op实现时的Kernel函数调用。

## Op

```cpp
using AttributeMap = Map<std::string, boost::any>;
enum DeviceType {
  kDEVICE_CPU = 0,
  kDEVICE_GPU,
  kNUM_DEVICES
};
using ShapeInfererFN = std::function<void(const Vec<TensorAttrPtr>& inputs, 
                                          const Vec<TensorAttrPtr>& outputs,
                                          const AttributeMap& attrs)>;
using GetGraidentFN = std::function<Vec<Op>(
    const Vec<TensorAttrPtr>& I,
    const Vec<TensorAttrPtr>& O,
    const Vec<TensorAttrPtr>& OG,
    const Vec<TensorAttrPtr>& IG,
    const AttributeMap& attrs,
)>;

using KernelFN = std::function<void(const Vec<Tensor>& inputs, 
                                    const Vec<Tensor>& outputs,
                                    const AttributeMap& attrs)>;

struct OpMeta {
  std::string type_;
  ShapeInfererFN shape_;
  GetGradientFN grad_;
  KernelFN kernels_[kNUM_DEVICES];
};

struct Op {
  std::string type_;
  Vec<TensorAttrPtr> inputs_;
  Vec<TensorAttrPtr> outputs_;
  AttributeMap attrs_;
};
```

`Op`是每一个操作的具体配置，而`OpMeta`是每一类操作的元信息，他们之间通过共同的`type_`来相互对应。用户配置Op的时候，将对应的Op创建，添加进Graph中即可。

每一个Op具有一些可配置的属性，这些可配置的属性是`AttributeMap`类型，即`Map<std::string, boost::any>`类型。该类型可以方便用户对一个Op配置float，string，int等不同类型的属性。

`OpMeta`类型即为每一个Op的元信息，里面包括了

* shape. 不同shape的输入，经过这个Op后会产生输出的shape。在这个函数中，可以通过throw exception报错。
* grad. 某一个Op对应的梯度Op是哪些。grad可以为空。为空表示这个Op不支持反向传播。
* kernels。 不同设备上如何计算该Op。

简单的实现一个Op为:

```
static void FCShape(const Vec<TensorAttrPtr>& inputs, 
                    const Vec<TensorAttrPtr>& outputs,
                    const AttributeMap& attrs) {
  outputs[0]->dims = {inputs[0]->dims[0], inputs[1]->dims[1]};                    
}

static void FCImpl(const Vec<Tensor>& inputs, 
                   const Vec<Tensor>& outputs,
                   const AttributeMap& attrs) {
  ...                   
}

static void Vec<Op> FCGrad(
    const Vec<TensorAttrPtr>& I,
    const Vec<TensorAttrPtr>& O,
    const Vec<TensorAttrPtr>& OG,
    const Vec<TensorAttrPtr>& IG,
    const AttributeMap& attrs,
) {
  Op op;
  op.type = "fc_grad";
  op.inputs = {I[0], I[1], OG[0]};
  op.outputs = {IG[1], IG[0], IG[2]};
  return {op};
}

static InitFunction init([]{
  OpMeta meta;
  meta.type_ = "fc";
  meta.shape_ = FCShape;
  meta.kernels[CPU] = FCImpl;
  meta.grad_ = FCGrad;
  OpMeta::addMeta(meta);
});
```
