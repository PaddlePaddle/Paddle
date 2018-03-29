# How to use Eigen in Paddle

Essentially, a neural network is a compute graph. T data needed for the computation is stored in `Tensor`s and its computation procedure is described by `Operator`s. An `Operator` calls the `Compute` interface in its corresponding `OpKernel` and operates on the `Tensor`.


## Eigen Tensor Module

The Eigen Tensor module supports powerful element-wise computation. In addition, a piece of code written using it can be run on both the CPU and the GPU.

Note that Eigen Tensor is still being actively developed, so its tests are not completely covered and its documentation may be sparse.

For details on Eigen Tensor module, please see [doc 1](https://github.com/RLovelett/eigen/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md) and [doc 2](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md).


## paddle::framework::Tensor

Paddle Tensor's is defined in the framework directory with the following interface:

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

`Placeholder` is used to delay memory allocation; that is, we can first define a tensor, using `Resize` to configure its shape, and then call `mutuable_data` to allocate the actual memory.

```cpp
paddle::framework::Tensor t;
paddle::platform::CPUPlace place;
// set size first
t.Resize({2, 3});
// allocate memory on CPU later
t.mutable_data(place);
```

### paddle::framework::Tensor Usage
`AddOp` demonstrates Tensor's usage.

- InferShape

When computing a neural network's compute graph, first call every `Operator`'s `InferShape` method, and use `Resize` to configure the size of the output tensor.

```cpp
void InferShape(const framework::InferShapeContext &ctx) const override {
  PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->dims(),
                    ctx.Input<Tensor>("Y")->dims(),
                    "Two input of Add Op's dimension must be same.");
  ctx.Output<Tensor>("Out")->Resize(ctx.Input<Tensor>("X")->dims());
}
```


- Run

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


## paddle::framework::Tensor到EigenTensor的转换

As shown above, in actual computation, we need to transform the input and output `Tensor`s into formats Eigen supports. We show some functions in [eigen.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/eigen.h) to implement the transformation from `paddle::framework::Tensor`to `EigenTensor/EigenMatrix/EigenVector/EigenScalar`.

Using EigenTensor as an example:

```cpp
Tensor t;
float* p = t.mutable_data<float>(make_ddim({1, 2, 3}), platform::CPUPlace());
for (int i = 0; i < 1 * 2 * 3; i++) {
  p[i] = static_cast<float>(i);
}

EigenTensor<float, 3>::Type et = EigenTensor<float, 3>::From(t);
```

`From` is an interfacing method provided by the EigenTensor template, which implements the transformation from a `paddle::framework::Tensor` object to an EigenTensor. Since `rank` is a template parameter, it needs to be explicitly specified at the time of the transformation.

In Eigen, tensors with different ranks are different types, with `Vector` bring a rank-1 instance. Note that `EigenVector<T>::From` uses a transformation from an 1-dimensional Paddle tensor to a 1-dimensional Eigen tensor while `EigenVector<T>::Flatten` reshapes a paddle tensor and flattens it into a 1-dimensional Eigen tensor. Both resulting tensors are still typed EigenVector.

For more transformations, see the [unit tests](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/eigen_test.cc) in the `eigen_test.cc` file.



## Implementing Computation

While computing, the device interface is needed from the EigenTensors on the left hand side of the assignments. Note that the computation between EigenTensors only changes the data originally inthe Tensor and does not change all the shape information associated with the Tensor.

```cpp
auto x = EigenVector<T>::Flatten(*input0);
auto y = EigenVector<T>::Flatten(*input1);
auto z = EigenVector<T>::Flatten(*output);
auto place = context.GetEigenDevice<Place>();
z.device(place) = x + y;
```

In this code segment, input0/input1/output can be Tensors of arbitrary dimension. We are calling Flatten from EigenVector, transforming a tensor of any dimension into a 1-dimensional EigenVector. After completing computation, input0/input1/output will retain the same shape information, and they can be resized using the `Resize` interface.

Because the Eigen Tensor module is under-documented, please refer to `OpKernel`'s computation code in TensorFlow's [kernel module documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels).
