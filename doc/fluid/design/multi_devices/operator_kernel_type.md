# Design Doc: The Keys of Operator Kernel Type
## Problem
An operator can have different kernel implementations, and each operator will have a map to store the related kernels. Fluid uses `OpKernelType` as a key to identify a unique kernel. Before an operator runs, a certain type of kernel must be chosen via a key of `OpKernelType`. Currently, `OpKernelType` is defined as follows:

```cpp
struct OpKernelType {
  platform::Place place_;
  proto::DataType data_type_;
};
```
For more details, please refer to [codes](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L348-L374) in github.

It contains two keys, `Place` and `DataType`. And these two keys will be hashed to a unique key to represent a certain type of kernel. However, these two keys do not provide enough information. We need a more complete representation of `OpKernelType`.

We often implement a kernel of an operator with some computing library on certain device(place). Please note that computing library and device do not have a one-to-one correspondence. A device can have a lot of computing libraries and a computing library can also support different devices.

For example, Eigen library supports Nvidia GPU/AMD GPU/CPU and MKLDNN library supports Intel CPU/Intel FPGA. Both `Place` and `Library` should be a key of `OpKernelType`.

Different DataTypes, such as fp64/fp32/int8, will obviously have different kernels. But different data layout of a Tensor will also lead to different implementations. Please refer to the batch norm operator [kernels](https://github.com/PaddlePaddle/Paddle/blob/a948fac4d0ad7e0412d373b8aabeb711c2899563/paddle/operators/batch_norm_op.cc#L180-L209) as an example. Data layout should also be taken into consideration.

## Solution

There are four keys to determine a kernel type of an operator: `Place`/`Library`/`DataType`/`Layout`.

```cpp
struct OpKernelType {
  platform::Place place_;
  platform::Library library_;
  proto::DataType data_type_;
  framework::Layout layout_;
};
```

The details are as follows:

### Place

`Place` is defined as:

```cpp
typedef boost::variant<CUDAPlace, ROCmPlace, FPGAPlace, CPUPlace> Place;
```

`Place` represents the device memory where data is located.


### Library

One operator kernel is usually implemented based on one library. `Library` is defined as a enum variable:

```cpp
enum Library { Plain, MKLDNN, CUDNN };
```

We use `Plain` enumerator to represent default library. Since most operators in Fluid are implemented based on the `Eigen` library, we take `Eigen` library as the `Plain` enumerator.
A library usually has a corresponding `DeviceContext` which contains some handles needed for computation. Fluid now has two default DeviceContexts for CPU and CUDA, namely, `CPUDeviceContext` and `CUDADeviceContext`. `CPUDeviceContext` contains an Eigen library handle and `CDUADeviceContext` contains an Eigen library handle and a cuBLAS handle.

If we want to support new library, a new enumerator need to be added to `Library` and a corresponding new `LibraryDeviceContext` need to be created.


### DataType


`DataType` is defined in [framework.proto](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto). Currently, int32/int64/fp32/fp64 are supported.

### Layout

Actually, a Tensor is a view of a block of memory. Besides a pointer to the memory, we also have to get some other descriptions of this block of memory, such as shape(ddim), stride, and layout.

Different layout leads to different implementation of the operator kernel. There are mainly 4 principles we have to follow to support layout in our Fluid framework.

- We take layout as a data member of Tensor. Layout is actually a enum variable. If Fluid is built with MKLDNN, then the memory format in MKLDNN will also be added into this enum variable.

- Users have to set layout for input data. And some operators like fill_constant/random, also have to set layout for generating data. Of course, we can have some default layout, like NCHW.

- The inference of Layout is at run-time, not at compile-time.

- Every operator has to implement different kernels for different layouts. Let's take MKLDNN as an example. If we want to implement an MKLDNN convolution operator, we have to implement all the kernels for different layouts, which are listed [here](http://intel.github.io/mkl-dnn/structmkldnn_1_1memory.html). And we will have a special macro to  register kernels for MKLDNN operators.

`Layout` is also defined as a enum variable:

```cpp
enum Layout {
  kNCHW,
  kNHWC,
#ifdef PADDLE_WITH_MKLDNN
  knChw8c
  ...
#endif
};
```
