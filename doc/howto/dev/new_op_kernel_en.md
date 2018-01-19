## How to Add OpKernels for new Device or Library

### Background

- A detailed documentation of how to add new operator and kernel is here: [`new_op_and_kernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/dev/new_op_en.md)
- we use `OpKernelType` to describe the attribute of each kernel. The design is [`op_kernel_type`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/operator_kernel_type.md)
- The mechanism that an Operator choose a kernel is described in this document: [`switch_kernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/switch_kernel.md)

### Write OpKernel for new Device or Library

#### Add new [library](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/library_type.h#L24)
If you have a new kind of library, such as MKLDNN, you need to add a new library_type. Now we have:

```
enum class LibraryType {
  kPlain = 0,
  kMKLDNN = 1,
  kCUDNN = 2,
};
```


#### Add new [place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h#L53)
If you have a new kind of Device, firstly you need to add a new kind of [`Place`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h#L53). For example `CUDAPlace`:

```cpp
struct CUDAPlace {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  // needed for variant equality comparison
  inline bool operator==(const CUDAPlace &o) const {
    return device == o.device;
  }
  inline bool operator!=(const CUDAPlace &o) const { return !(*this == o); }

  int device;
};

typedef boost::variant<CUDAPlace, CPUPlace> Place;
```

#### Add [device context]((https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/device_context.h#L37))
After a new kind of Device is added, you should add a correspodding [DeviceContext](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/device_context.h#L37) for it.

```cpp
class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  virtual void Wait() const {}
};
```

#### 3. implement new [OpKernel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/operator.h#L351) for your Device.

A detailed documentation can be found in [`new_op_and_kernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/dev/new_op_en.md)

```cpp
class OpKernelBase {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernelBase() = default;
};

template <typename T>
class OpKernel : public OpKernelBase {
 public:
  using ELEMENT_TYPE = T;
};
```


#### 4. Register the OpKernel to framework

After writing the components described above, we should register the kernel to the framework.

We use `REGISTER_OP_KERNEL` to do the registration.

```cpp
REGISTER_OP_KERNEL(
	op_type, 
	library_type, 
	place_type, 
	kernel0, kernel1, ...)
```

take [`conv2d`]((https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/conv_cudnn_op.cu.cc#L318)) as an example:

	```cpp
	REGISTER_OP_KERNEL(conv2d, CUDNN, ::paddle::platform::CUDAPlace,
	                   paddle::operators::CUDNNConvOpKernel<float>,
	                   paddle::operators::CUDNNConvOpKernel<double>);
	```

In the code above:

 - `conv2d` is the type/name of the operator
 - `CUDNN` is `library`
 - `::paddle::platform::CUDAPlace` is `place`
 - template parameter `float/double` on `CUDNNConvOpKernel<T>` is `data_type`.
