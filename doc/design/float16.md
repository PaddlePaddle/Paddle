# Design Doc: float16

## Why float16
Half precision (float16) is a binary floating-point format that occupies 16 bits in memory. float16 is half the size of traditional 32-bit single precision format (float) and has lower precision and smaller range. 

When high precision computation is not required, using float16 data type could potentially 

- reduce storage space, memory bandwidth, and power usages; 
- increase the chance of data fitting into a smaller cache of lower latency; 
- provide arithmetic speed up if supported by hardware. 

## Survey of current float16 support
A brief survey of float16 support on different compilers, hardwares, and libraries can be found below. Interested readers can refer to [link1](https://github.com/PaddlePaddle/Paddle/issues/4853) and [link2](https://github.com/Xreki/Xreki.github.io/blob/master/multi_data_types_in_dl_framework/ppt/float16_and_quantized_type.md) for more info.

The goal of float16 is to serve as a key for the executor to find and run the correct version of compute method specialized for float16 in operator kernel. It should be compatible with various natively supported float16 implementations including `__half` for cuda, `float16_t` for ARM, and `Eigen::half` for Eigen to make writing customized float16 kernels easier. 

### Compiler
- nvcc supports `__half` data type after CUDA 7.5.
- `__fp16` or `float16_t` is supported as storage type for gcc >= 6.1 and clang >= 3.4.
- `__fp16` or `float16_t` is supported as arithmetic type for gcc >= 7.1 and clang >= 3.9.

### Hardware
- `__half` is supported on GPU with compute capability >= 5.3.
- `__fp16` is supported as storage type for ARMv7-A, ARMv8-A, and above.
- `__fp16` is supported as arithmetic type after ARMv8.2-A (currently, the only microarchitecture implementing ARMv8.2-A is ARM Cortex-A75, which is announced in May 2017. There seems to be no application processors currently available on market that adopts this architecture. It is reported that Qualcomm Snapdragon 845 uses Cortex-A75 design and will be available in mobile devices in early 2018).

### Libraries
- [Eigen](https://github.com/RLovelett/eigen) >= 3.3 supports float16 calculation on both GPU and CPU using the `Eigen::half` class. It is mostly useful for Nvidia GPUs because of the overloaded arithmetic operators using cuda intrinsics. It falls back to using software emulation on CPU for calculation and there is no special treatment to ARM processors.
- [ARM compute library](https://github.com/ARM-software/ComputeLibrary) >= 17.02.01 supports NEON FP16 kernels (requires ARMv8.2-A CPU).


## Implementation
The float16 class holds a 16-bit `uint16_t` data internally.
```
struct float16 {
  uint16_t x;
};
``` 

float16 supports the following features:
  - constructors / assignment operators that take input from primitive data types including bool, integers of various length, float, and double. 
  - constructors / assignment operators that take input from `__half` on cuda, `float16_t` on ARM, and `Eigen::half` on Eigen.
  - conversion operators to primitive data types and half precision data types on cuda, ARM and Eigen. 
  - overloaded arithmetic operators for cuda, arm, and non-arm cpu, respectively. These operators will take advantage of the cuda and ARM intrinsics on the corresponding hardware. 
  
To support the above features, two fundamental conversion functions are provided:
```
float16 float_to_half_rn(float f);  // convert to half precision in round-to-nearest-even mode
float half_to_float(float16 h);
```
which provides one-to-one conversion between float32 and float16. These twos functions will do different conversion routines based on the current hardware. CUDA/ARM instrinsics will be used when the corresonding hardware is available. If the hardware or compiler level does not support float32 to float16 conversion, software emulation will be performed to do the conversion.

## To do
After float16 class is available, some of the future items are below:

- Update pybind/tensor_py.h to bind c++ float16 with numpy float16. 

- Modify `GetKernelType()` method in `framework/operator.h` to make it compatible with float16.

- Create a type-casting operator that can convert the data type in tensor between float16 and other types.
