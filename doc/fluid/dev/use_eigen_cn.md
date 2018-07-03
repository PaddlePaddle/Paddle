# 在Paddle中如何使用Eigen

神经网络本质上是一个计算图，计算需要的数据存放在`Tensor`中，而计算过程是由`Operartor`来描述的。在执行时，`Operator`调用对应`OpKernel`中的`Compute`接口，实现对`Tensor`的操作。


## Eigen Tensor模块

Eigen Tensor模块对element-wise计算提供了强大的支持，并且书写一份代码，可以同时在CPU、GPU执行。但Eigen Tensor是一个正在开发中的模块，因此可能测试不够完备，文档较少。

关于Eigen Tensor模块的详细介绍请参考[文档1](https://github.com/RLovelett/eigen/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md) 和[文档2](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md)


## paddle::framework::Tensor

Paddle Tensor定义在framework目录下，其主要接口如下：

```cpp
class Tensor {
 public:
  /*! Return a pointer to mutable memory block. */
  template <typename T>
  inline T* data();

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(platform::Place place);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims    The dimensions of the memory block.
   * @param[in] place   The place of the memory block.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(DDim dims, platform::Place place);

  /*! Resize the dimensions of the memory block. */
  inline Tensor& Resize(const DDim& dims);

  /*! Return the dimensions of the memory block. */
  inline const DDim& dims() const;

 private:  
  /*! holds the memory block if allocated. */
  std::shared_ptr<Placeholder> holder_;

  /*! points to dimensions of memory block. */
  DDim dim_;
};
```

`Placeholder`的作用是延迟分配内存，即我们可以先定义一个Tensor，然后使用Resize接口设置Tensor的大小，最后再调用mutable_data接口分配实际的内存。

```cpp
paddle::framework::Tensor t;
paddle::platform::CPUPlace place;
// set size first
t.Resize({2, 3});
// allocate memory on CPU later
t.mutable_data(place);
```

### paddle::framework::Tensor使用样例
下面以AddOp为例说明Tensor的使用过程：

- InferShape

在运行神经网络计算图时，我们先调用每个`Operator`的`InferShape`接口，根据输入Tensor的大小来设置输出Tensor的大小，`Resize`接口会被调用。

```cpp
void InferShape(const framework::InferShapeContext &ctx) const override {
  PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->dims(),
                    ctx.Input<Tensor>("Y")->dims(),
                    "Two input of Add Op's dimension must be same.");
  ctx.Output<Tensor>("Out")->Resize(ctx.Input<Tensor>("X")->dims());
}
```


- Run

`Operator`的`Run`接口最终会调用对应`OpKernel`的`Compute`接口，在这时真正的分配内存，`mutable_data`接口会被调用。

```cpp
void Compute(const framework::ExecutionContext& context) const override {
  auto* input0 = context.Input<Tensor>("X");
  auto* input1 = context.Input<Tensor>("Y");
  auto* output = context.Output<Tensor>("Out");

  output->mutable_data<T>(context.GetPlace());

  auto x = EigenVector<T>::Flatten(*input0);
  auto y = EigenVector<T>::Flatten(*input1);
  auto z = EigenVector<T>::Flatten(*output);

  auto place = context.GetEigenDevice<Place>();

  z.device(place) = x + y;
}
```


### paddle::framework::Tensor到EigenTensor的转换

如上一小节所示，在具体的计算中，我们需要先把输入Tensor和输出Tensor转换为Eigen支持的格式。我们在[eigen.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/eigen.h)中提供了一些全局函数用来实现paddle::framework::Tensor到EigenTensor/EigenMatrix/EigenVector/EigenScalar的转换。

以EigenTensor为例，做一个介绍

```cpp
Tensor t;
float* p = t.mutable_data<float>(make_ddim({1, 2, 3}), platform::CPUPlace());
for (int i = 0; i < 1 * 2 * 3; i++) {
  p[i] = static_cast<float>(i);
}

EigenTensor<float, 3>::Type et = EigenTensor<float, 3>::From(t);
```

From是EigenTensor模板提供的一个接口，可以实现从paddle::framework::Tensor到对EigenTensor的转换。由于Tensor的rank是模板参数，因此在转换时需要显示的指定。

在Eigen中，不同rank的Tensor是不同类型，Vector是rank为1的Tensor。需要额外注意的是，EigenVector<T>::From方法是把paddle中的一维Tensor转为Eigen的一维Tensor，在这里用EigenVector来表示；而EigenVector<T>::Flatten方法是把paddle中的一个Tensor进行reshape操作，压扁成为Eigen的一维Tensor，类型仍然为EigenVector。

更多的转换方法请参考eigen_test.cc中的[单元测试](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/eigen_test.cc)。



## 实现计算

当需要完成计算时，我们需要等式左边的EigenTensor调用device接口。在这里需要注意的是，这里的EigenTensor之间的运算只是改变了原有Tensor中的数据，而不会改变原有Tensor的shape信息。

```cpp
auto x = EigenVector<T>::Flatten(*input0);
auto y = EigenVector<T>::Flatten(*input1);
auto z = EigenVector<T>::Flatten(*output);
auto place = context.GetEigenDevice<Place>();
z.device(place) = x + y;
```

在这段代码中，input0/input1/output可以是任意维度的Tensor。我们调用了EigenVector的Flatten接口，把任意维度的Tensor转为了一维的EigenVector。而在计算结束之后，input0/input1/output的原有shape信息不变。如果想改变原有Tensor的shape信息，可以调用Resize接口进行改变。

由于Eigen Tensor模块的文档较少，我们可以参考TensorFlow的[kernels](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels)模块下的相关`OpKernel`的计算代码。
