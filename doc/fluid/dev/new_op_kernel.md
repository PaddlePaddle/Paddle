# Add Kernels for a New Device

## Background

PaddlePaddle Fluid have hundreds of operators.  Each operator could have one or more kernels.  A kernel is an implementation of the operator for a certain device, which could be a hardware device, e.g., the CUDA GPU, or a library that utilizes a device, e.g., Intel MKL that makes full use of the Xeon CPU.

[This document](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/new_op_en.md) explains how to add an operator, and its kernels.  The kernels of an operator are indexed by a C++ type [`OpKernelType`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/multi_devices/operator_kernel_type.md).  An operator chooses the right kernel at runtime.  This choosing mechanism is described [here](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/execution/switch.md).

## Write Kernels for A New Device

### Add A New Device

  For some historical reaons, we misuse the word *library* for *device*.  For example, we call the deivce type by *library type*.  An example is the header file [`library_type.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/library_type.h#L24).  We will correct this ASAP.

To register a new device, we need to add an enum value to `LibraryType`:

```
enum class LibraryType {
  kPlain = 0,
  kMKLDNN = 1,
  kCUDNN = 2,
};
```


### Add A New [Place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/place.h#L53)

If you have a new kind of Device, firstly you need to add a new kind of [`Place`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/place.h#L53). For example `CUDAPlace`:

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

### Add [device context]((https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/device_context.h#L37))
After a new kind of Device is added, you should add a corresponding [DeviceContext](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/device_context.h#L37) for it.

```cpp
class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  virtual void Wait() const {}
};
```

### Implement new [OpKernel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L351) for your Device.

A detailed documentation can be found in [`new_op_and_kernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/new_op_en.md)

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


### Register the OpKernel to framework

After writing the components described above, we should register the kernel to the framework.

We use `REGISTER_OP_KERNEL` to do the registration.

```cpp
REGISTER_OP_KERNEL(
	op_type,
	library_type,
	place_type,
	kernel0, kernel1, ...)
```

kernel0, kernel1 are kernels that have the same `op_type`, `library_type`, `place_type` but different `data_types`.

take [`conv2d`]((https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/conv_cudnn_op.cu.cc#L318)) as an example:

	```cpp
	REGISTER_OP_KERNEL(conv2d, CPU, paddle::platform::CPUPlace,
    		paddle::operators::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    		paddle::operators::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);

	REGISTER_OP_KERNEL(conv2d, CUDNN, ::paddle::platform::CUDAPlace,
	       paddle::operators::CUDNNConvOpKernel<float>,
	       paddle::operators::CUDNNConvOpKernel<double>);
	```

In the code above:

 - `conv2d` is the type/name of the operator
 - `CUDNN/CPU` is `library`
 - `paddle::platform::CUDAPlace/CPUPlace` is `place`
 - template parameter `float/double` on `CUDNNConvOpKernel<T>` is `data_type`.
