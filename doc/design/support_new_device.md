# Design Doc: Support new Device/Library

## Background

Deep learning has a high demand for computing resources. New high-performance device and computing library are coming constantly. The deep learning framework has to integrate these high-performance device and computing library flexibly.

On the one hand, hardware and computing library are not usually one-to-one coresponding relations. For example, in Intel CPU, there are Eigen and MKL computing library. And in Nvidia GPU, there are Eigen and cuDNN computing library. We have to implement specific kernels for an operator for each computing library.

On the other hand, users usually do not want to care about the low-level hardware and computing library when writing a neural network configuration. In Fluid, `Layer` is exposed in `Python`, and `Operator` is exposed in `C++`. Both `Layer` and `Operator` are independent on hardwares.

So, how to support a new Device/Library in Fluid becomes a challenge.


## Basic: Integrate A New Device/Library

For a general overview of fluid, please refer to [overview doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/read_source.md).

There are mainly there parts we have to consider in integrating a new device/library:

- Place and DeviceContext: indicates the device id and manages hardware resources

- Memory and Tensor: malloc/free data on certain device

- Math Functor and OpKernel: implement computing unit on certain device/library

### Place and DeviceContext


#### Place
Fluid use class [Place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h#L55) to represent specific device and computing library. There are inheritance relationships between different kinds of `Place`.

```
        |   CPUPlace   --> MKLDNNPlace
Place --|   CUDAPlace  --> CUDNNPlace
        |   FPGAPlace
```

And `Place` is defined as follows:

```
typedef boost::variant<CUDAPlace, CPUPlace, FPGAPlace> Place;
```

#### DeviceContext

Fluid use class [DeviceContext](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/device_context.h#L30) to manage the resources in certain hardware, such as CUDA stream in `CDUADeviceContext`. There are also inheritance relationships between different kinds of `DeviceContext`.


```
                /->  CPUDeviceContext   --> MKLDeviceContext
DeviceContext ---->  CUDADeviceContext  --> CUDNNDeviceContext
                \->  FPGADeviceContext
```

A example of Nvidia GPU is as follows:

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

- CUDNNDeviceContext

```
class CUDNNDeviceContext : public CUDADeviceContext {
  private:
    cudnnHandle_t cudnn_handle_;
};
```


### Memory and Tensor


#### memory module

Fluid provide following [memory interfaces](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/memory/memory.h#L36):

```
template <typename Place>
void* Alloc(Place place, size_t size);

template <typename Place>
void Free(Place place, void* ptr);

template <typename Place>
size_t Used(Place place);
```

To implementing these interfaces, we have to implement MemoryAllocator for specific Device


#### Tensor

[Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/tensor.h#L36) holds data with some shape in certain Place.

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



### Math Functor and OpKernel

Fluid implements computing unit based on different DeviceContext. Some computing unit is shared between operators. These common part will be put in operators/math directory as basic Functors.

Let's take [MaxOutFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/math/maxouting.h#L27) as an example:

The interface is defined in header file.

```
template <typename DeviceContext, typename T>
class MaxOutFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* output, int groups);
};
```

CPU implement in .cc file

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

CUDA implement in .cu file

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


We get computing handle from concrete DeviceContext, and make compution on tensors.

The implement of `OpKernel` is similar to math functors, the extra thing we need to do is registering the OpKernel to global map.

Fluid provides different register interface in op_registry.h


Let's take [Crop](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/crop_op.cc#L134) operator as an example:

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

Generally, we will impelement OpKernel for all Device/Library of an Operator. We can easily train a Convolutional Neural Network in GPU. However, some OpKernel is not sutibale in a specific Device. For example, crf operator can be only run at CPU, whereas most other operators can be run at GPU. To achieve high performance in such circumstance, we have to switch between different Device/Library.


We will discuss how to implement an efficient OpKernel switch policy. 

- TBD
