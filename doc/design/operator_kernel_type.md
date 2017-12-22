# Design Doc: The Keys of Operator Kernel Type
## Problem
An operator can have different kernel implementations, and each operator will have a map to store the related kernels. Fluid uses `OpKernelType` as a key to identify a unique Kernel. Before an operator runs, an certain kernel must be chosen by a key of `OpKernelType`. Currently, `OpKernelType` is defined as follows:

```cpp
struct OpKernelType {
  platform::Place place_;
  proto::DataType data_type_;
};
```
For more details, please refer to [codes](https://github.com/PaddlePaddle/Paddle/blob/2d5ec16bc8a09fb8e0f62c89b116b0cd1d333907/paddle/framework/operator.h#L348-L374) in github.

It contains two keys, `Place` and `DataType`. And these two keys will be hashed to a unique key to represent a certain type of kernel. However, these two keys are not enough. We need a more complete representation of `OpKernelType`. 

We often implement a kernel of an operator with some computing library in certain device(place). Please remind that computing library and device are not one-to-one corresponding. A device can have a lot of computing libraries and a computing library can also support several devices. 

For example, Eigen library can support Nvidia GPU/AMD GPU/CPU. And MKLDNN library can support Intel CPU/Intel FPGA. Both `Place` and `Library` should be a key of `OpKernelType`.

It's obvious that different DataTypes, like fp64/fp32/int8 will have different kernels. But the data layout of a Tensor will also lead to different implementation. Please refer to the batch norm operator [kernels](https://github.com/PaddlePaddle/Paddle/blob/a948fac4d0ad7e0412d373b8aabeb711c2899563/paddle/operators/batch_norm_op.cc#L180-L209). Data Layout should also be taken into consideration.

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

Following is the details:

### Place

`Place` is defined as follows:

```cpp
typedef boost::variant<CUDAPlace, ROCmPlace, FPGAPlace, CPUPlace> Place;
```

`Place` is to represent the device memory where data is locating.


### Library

One operator kernel is usually implemented based on one library. `Library` is defined as a enum variable:

```cpp
enum Library { Plain, MKLDNN, CUDNN };
```

We use `Plain` enumerator to represent default library. Since most operators in Fluid are implemented based on `Eigen` library, we take `Eigen` library as the `Plain` enumerator.
A library usually has a corresponding `DeviceContext` which contains some handles needed by computation. Fluid now have two default DeviceContexts in CPU and CUDA, `CPUDeviceContext` and `CUDADeviceContext`. `CPUDeviceContext` contains a Eigen library handle and `CDUADeviceContext` contains a Eigen library handle and cuBLAS handle.

If we want to support new Library, a new enumerator need to be added to `Library` and a new corresponding `LibraryDeviceContext` will be created.


### DataType


`DataType` is defined in [framework.proto](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto). Currently, int32/int64/fp32/fp64 are supported.

### Layout

Actually, a Tensor is a view of a block of memory. Besides a pointer to the memory, we also have to get some other descriptions of this block of memory, such as shape(ddim), stride, and layout.

Different layout leads to different implementation of operator kernel. There are mainly 4 principles we have to follow to support layout in our fluid framework.

- We take layout as a data member of Tensor. Layout is actually a enum variable. If fluid is built with MKLDNN, then, the memory format in MKLDNN will be added into this enum variable too.

- Users have to set layout for input data. And some operators like fill_constant/random, also have to set layout of generating data. Of course, we can have some default layout, like NCHW.

- The inference of Layout is at run-time, not compile-time.

- Every operator have to implement different kernels for different layouts. Let's take MKLDNN as an example, if we want to implement a MKLDNN convolution operator, we have to realize all the kernels for different layout, list at [here](http://01org.github.io/mkl-dnn/structmkldnn_1_1memory.html). And we will have a special macro to do registering kernels for MKLDNN operators.

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
