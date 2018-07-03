# Design Doc: Supporting new Device/Library

## Background

Deep learning has a high demand for computing resources. New high-performance devices and computing libraries are appearing very frequently. Deep learning frameworks have to integrate these high-performance devices and computing libraries in a flexible and efficient manner.

On one hand, hardware and computing libraries usually do not have a one-to-one correspondence. For example, Intel CPUs support Eigen and MKL computing libraries while Nvidia GPUs support Eigen and cuDNN computing libraries. We have to implement operator specific kernels for each computing library.

On the other hand, users usually do not want to care about the low-level hardware and computing libraries when writing a neural network configuration. In Fluid, `Layer` is exposed in `Python`, and `Operator` is exposed in `C++`. Both `Layer` and `Operator` are hardware independent.

So, how to support a new Device/Library in Fluid becomes a challenge.


## Basic: Integrate A New Device/Library

For a general overview of fluid, please refer to the [overview doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/read_source.md).

There are mainly three parts that we have to consider while integrating a new device/library:

- Place and DeviceContext: indicate the device id and manage hardware resources

- Memory and Tensor: malloc/free data on certain device

- Math Functor and OpKernel: implement computing unit on certain devices/libraries

### Place and DeviceContext

Please note that device and computing library are not one-to-one corresponding. A device can have a lot of computing libraries and a computing library can also support several devices.

#### Place
Fluid uses class [Place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/place.h#L55) to represent the device memory where data is located. If we add another device, we have to add the corresponding `DevicePlace`.

```
        |   CPUPlace
Place --|   CUDAPlace
        |   FPGAPlace
```

And `Place` is defined as follows:

```
typedef boost::variant<CUDAPlace, CPUPlace, FPGAPlace> Place;
```

#### DeviceContext

Fluid uses class [DeviceContext](https://github.com/PaddlePaddle/Paddle/blob/develop/fluid/paddle/platform/device_context.h#L30) to manage the resources in different libraries, such as CUDA stream in `CDUADeviceContext`. There are also inheritance relationships between different kinds of `DeviceContext`.


```
                /->  CPUDeviceContext   
DeviceContext ---->  CUDADeviceContext  
                \->  FPGADeviceContext
```

An example of Nvidia GPU is as follows:

- DeviceContext


```
class DeviceContext {
  virtual Place GetPlace() const = 0;
};  
```


- CUDADeviceContext


```
class CUDADeviceContext : public DeviceContext {
  Place GetPlace() const override { return place_; }
private:
  CUDAPlace place_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
  std::unique_ptr<Eigen::GpuDevice> eigen_device_;  // binds with stream_
};
```

### Memory and Tensor


#### memory module

Fluid provides the following [memory interfaces](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/memory/memory.h#L36):

```
template <typename Place>
void* Alloc(Place place, size_t size);

template <typename Place>
void Free(Place place, void* ptr);

template <typename Place>
size_t Used(Place place);
```

To implement these interfaces, we have to implement MemoryAllocator for different Devices.


#### Tensor

[Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/tensor.h#L36) holds data with some shape in a specific Place.

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

`Placeholder` is used to delay memory allocation; that is, we can first define a tensor, using `Resize` to configurate its shape, and then call `mutuable_data` to allocate the actual memory.

```cpp
paddle::framework::Tensor t;
paddle::platform::CPUPlace place;
// set size first
t.Resize({2, 3});
// allocate memory on CPU later
t.mutable_data(place);
```



### Math Functor and OpKernel

Fluid implements computing units based on different DeviceContexts. Some computing units are shared between operators. This common part will be put in operators/math directory as basic Functors.

Let's take [MaxOutFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/math/maxouting.h#L27) as an example:

The interface is defined in the header file.

```
template <typename DeviceContext, typename T>
class MaxOutFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* output, int groups);
};
```

CPU implementation is in .cc file

```
template <typename T>
class MaxOutFunctor<platform::CPUDeviceContext, T> {
  public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* output,
                  int groups) {
                  ...
                  }
};
```

CUDA implementation is in .cu file

```
template <typename T>
class MaxOutFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* output,
                  int groups) {
                  ...
                  }
};                  
```


We first obtain the computing handle from a concrete DeviceContext and then compute on tensors.

The implementation of `OpKernel` is similar to math functors, the extra thing we need to do is to register the OpKernel in a global map.

Fluid provides different register interfaces in op_registry.h


Let's take [Crop](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/crop_op.cc#L134) operator as an example:

In .cc file:

```
REGISTER_OP_CPU_KERNEL(crop, ops::CropKernel<float>);
REGISTER_OP_CPU_KERNEL(
    crop_grad, ops::CropGradKernel<paddle::platform::CPUDeviceContext, float>);
```

In .cu file:

```
REGISTER_OP_CUDA_KERNEL(crop, ops::CropKernel<float>);
REGISTER_OP_CUDA_KERNEL(
    crop_grad, ops::CropGradKernel<paddle::platform::CUDADeviceContext, float>);
```


## Advanced topics: How to switch between different Device/Library

Generally, we will implement OpKernel for all Device/Library of an Operator. We can easily train a Convolutional Neural Network in GPU. However, some OpKernel is not suitable on a specific Device. For example, crf operator can only run on CPU, whereas most other operators can run on GPU. To achieve high performance in such circumstance, we have to switch between different Device/Library.


For more details, please refer to following docs:

- operator kernel type [doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/multi_devices/operator_kernel_type.md)
- switch kernel [doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/execution/switch.md)
