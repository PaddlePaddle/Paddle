# Design Doc: float16

## Why float16
Half precision (float16) is a binary floating-point format that occupies 16 bits / 2 bytes in memory. float16 is half the size of traditional 32-bit single precision format (float) and has lower precision and smaller range. 

When high precision computation is not required, using float16 data type could potentially 

- reduce storage space, memory bandwidth, and power usages; 
- increase the chance of data fitting into a smaller cache of lower latency; 
- provide arithmetic speed up if supported by hardware. 

A brief survey of float16 support on different hardwares can be found [here](https://github.com/PaddlePaddle/Paddle/issues/4853). A brief survey of existing float16 implementations can be found [here](https://github.com/Xreki/Xreki.github.io/blob/master/multi_data_types_in_dl_framework/ppt/float16_and_quantized_type.md). 

There are various natively supported float16 implementations on different hardwares/linear algebra libraries including half on cuda, __fp16/float16_t on ARM processor, and Eigen::half on Eigen.

The goal of float16 is to serve as a key for the executor to find and run the correct version of operator kernel compute method specialized for float16. It should be compatible with half on cuda, __fp16 on ARM, and Eigen::half on Eigen to make writing customized float16 kernels easier. 

## Implementation
The float16 class holds a 2-byte uint16_t data internally.
```
struct float16 {
  uint16_t x;
};
``` 

float16 supports the following features:
  - constructors / assignment operators that take input from primitive data types including bool, integers of various length, float, and double. 
  - constructors / assignment operators that take input from half on cuda, __fp16 on ARM, and Eigen::half on Eigen.
  - conversion operators to primitive data types and half precision data types on cuda, ARM and Eigen. 
  - overloaded arithmetic operators (e.g., +, -, *, /) for cuda, arm, and non-arm cpu, respectively. These operators will take advantage of the cuda and ARM intrinsics on the corresponding hardware. 

To support the above features, two fundamental conversion functions are provided:
```
float16 float_to_half_rn(float f);  // convert to half precision in round-to-nearest-even mode
float half_to_float(float16 h);
```
which provides one-to-one conversion between float32 and float16. These twos functions will do different conversion routines based on the current hardware. CUDA/ARM instrinsics will be used when the corresonding hardware is available. When the hardware falls back to non-ARM cpu, software emulation will be performed to do the conversion.

## To do
After float16 class is available, some of the future items are below:

- Update pybind/tensor_py.h to bind c++ float16 with numpy float16. 

- Modify `IndicateDataType()` method in `framework/operator.h` to make it compatible with float16.

- Create a type-casting operator that can convert the data type in tensor between float16 and other types.
