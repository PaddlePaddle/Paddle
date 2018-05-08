# Design Doc: float16

## Why float16
Half precision (float16) is a binary floating-point format that occupies 16 bits in memory. float16 is half the size of traditional 32-bit single precision format (float) and has lower precision and smaller range. 

When high precision computation is not required (which is usually the case at least in the deep learning inference stage), using float16 data type could potentially 

- reduce storage space, memory bandwidth, and power usages; 
- increase the chance of data fitting into a smaller cache of lower latency; 
- provide arithmetic speed up if supported by hardware. 

## Survey of current float16 support
A brief survey of float16 support on different compilers, hardwares, and libraries can be found below. Interested readers can refer to [link1](https://github.com/PaddlePaddle/Paddle/issues/4853) and [link2](https://github.com/Xreki/Xreki.github.io/blob/master/multi_data_types_in_dl_framework/ppt/float16_and_quantized_type.md) for more info.

The goal of float16 is to serve as a key for the executor to find and run the correct version of compute method specialized for float16 in operator kernels. It should be compatible with various natively supported float16 implementations including `__half` for cuda, `float16_t` for ARM, and `Eigen::half` for Eigen to make writing customized float16 kernels easier. 

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

### CUDA version issue
There are currently three versions of CUDA that supports `__half` data type, namely, CUDA 7.5, 8.0, and 9.0. 
CUDA 7.5 and 8.0 define `__half` as a simple struct that has a `uint16_t` data (see [`cuda_fp16.h`](https://github.com/ptillet/isaac/blob/9212ab5a3ddbe48f30ef373f9c1fb546804c7a8c/include/isaac/external/CUDA/cuda_fp16.h)) as follows:
```
typedef struct __align__(2) {
   unsigned short x;
} __half;

typedef __half half;
```
This struct does not define any overloaded arithmetic operators. So you have to directly use `__hadd` instead of `+` to correctly add two half types:
```
__global__ void Add() {
  half a, b, c;
  c = __hadd(a, b); // correct
  c = a + b; // compiler error: no operator "+" matches these operands
}
```
CUDA 9.0 provides a major update to the half data type. The related code can be found in the updated [`cuda_fp16.h`](https://github.com/ptillet/isaac/blob/master/include/isaac/external/CUDA/cuda_fp16.h) and the newly added [`cuda_fp16.hpp`](https://github.com/ptillet/isaac/blob/master/include/isaac/external/CUDA/cuda_fp16.hpp).

Essentially, CUDA 9.0 renames the original `__half` type in 7.5 and 8.0 as `__half_raw`, and defines a new `__half` class type that has constructors, conversion operators, and also provides overloaded arithmetic operators such as follows:
```
typedef struct __CUDA_ALIGN__(2) {
    unsigned short x;
} __half_raw;


struct __CUDA_ALIGN__(2) __half {
protected:
    unsigned short __x;
public:
    // constructors and conversion operators from/to 
    // __half_raw and other built-in data types
}

typedef __half half;

__device__ __forceinline__ 
__half operator+(const __half &lh, const __half &rh) { 
    return __hadd(lh, rh); 
}

// Other overloaded operators
``` 
This new design makes `c = a + b` work correctly for CUDA half data type. 

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

## float16 inference
In Fluid, a neural network is represented as a protobuf message called [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md), whose Python wrapper is a [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#program). The basic structure of a program is some nested [blocks](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#block), where each block consists of some [variable](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#variable) definitions and a sequence of [operators](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#operator). An [executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/executor.md) will run a given program desc by executing the sequence of operators in the entrance block of the program one by one.  

### Operator level requirement
Each operator has many kernels for different data types, devices, and library types. The operator will select the appropriate kernel to run based on, among other things, the data type of the input variables. By default, every Fluid operator has a float data type kernel that takes float variables as input and generates float output. 

This means that if we provide float input to the first operator in a program, then each opeartor will use float kernel to compute float output and send it as input to the next operator to trigger the float kernel. Overall, the program will run in float mode and give us a final output of float data type.

The same principle applies if we want a program to run in float16 mode. We provide input variable of float16 data type to the first operator, and then one by one, each operator in the program will run the float16 kernel (provided that each operator in this program has float16 kernels registered) until we finally obtain a float16 output variable.

So the preliminary requirement for float16 inference is to add float16 kernel to operators that are needed in a specific kind of program. For example, float16 inference on an image classification neural network like Vgg or Resnet, typically requires the following operators to have float16 kernels: convolution, pooling, multiplication, addition, batch norm, dropout, relu, and softmax. Please refer to [new_op_en](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/new_op_en.md) for details of how to add new kernels to an operator.

### Variable level requirement
Operators including convolution and multiplication (used in fully-connected layers) takes as input not only the variables generated by the preceding operators but also [parameter](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#parameter) variables, which contains the trained weights to apply to the input data. These weights are obtained in the Fluid training process and are by default of float data type.

When these operators are running in float16 mode, the float16 kernel requires those parameter variables to contain weights of Fluid float16 data type. Thus, we need a convenient way to convert the original float weights to float16 weights. 

In Fluid, we use tensor to hold actual data for a variable on the c++ end. [Pybind](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/tensor_py.h) is used to bind c++ tensors of certain data type with numpy array of the correponding numpy data type on the Python end. Each common c++ built-in data type has a corresponding numpy data type of the same name. However, since there is no built-in float16 type in c++, we cannot directly bind numpy float16 data type with the Fluid float16 class. Since both Fluid float16 and numpy float16 use uint16 as the internal data storage type, we use c++ built-in type `uint16_t` and the corresponding numpy uint16 data type to bridge the gap via [Pybind](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/tensor_py.h). 

The following code demonstrates how to do the tensor conversion.
```Python
# var is the variable of float weights
# tensor is a numpy array of data copied from the tensor data in var 
# fp16_var is the variable that will contain float16 weights converted from var  
tensor = numpy.array(var.get_tensor())
fp16_tensor = fp16_var.get_tensor()

# After the original tensor data is converted to numpy float16 data type, 
# view(numpy.uint16) is used so that the internal memory of the numpy array 
# will be reinterpreted to be of uint16 data type, which is binded to 
# Fluid float16 class via pybind with the help of uint16_t built-in c++ type
fp16_tensor.set(tensor.astype(numpy.float16).view(numpy.uint16), GPUPlace)  
```

### Consistent API requirement
The basic inference in float16 mode requires users to feed input and obtain output both of float16 data type. However, in this way, the inference APIs are not consistent between float16 mode and float mode, and users may find it confusing and diffcult to use float16 inference since they need to do extra steps to provide float16 input data and convert float16 output data back to float. To have consistent API for different inference modes, we need to transpile the program desc in some way so that we can run float16 inference by feeding and fetching variables of float data type.

This problem can be solved by introducing a type-casting operator which takes an input variable of certain data type, cast it to another specified data type, and put the casted data into the output variable. Insert cast operator where needed can make a program internally run in float16 mode.   

### float16 transpiler
Put all the above requirements in mind, we designed a float16 inference transpiler that can tranpile a float32 mode inference program desc to a float16 mode one.

Given a float inference program and the corresponding variables of float32 weights in the [scope](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/scope.md),
this transpiler mainly does the following modifications:

1. Insert cast operators at the beginning of the program so that the input float data will be converted to float16 data type before feeding to subsequent operators to invoke the float16 kernel. 

2. Insert cast operators at the end of the program so that the output float16 data will be converted back to float data type before users obtain the result.

3. For each parameter variable of float weights, create in the scope a corresponding variable of float16 weights which are converted from the corresponding float weights and add this new float16 variable to the program.

4. Update the operator information in the program so that each relevant operator use the newly created float16 variable instead of its float counterpart.

Below is an example of usage:
```Python
# Get the float inference program
[float_inference_program, feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

# Prepare the float input data
tensor_img = numpy.random.rand(1, 3, 32, 32).astype(numpy.float32)

# Running inference_program in float mode
float_results = exe.run(float_inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets)

# Use float16 transpiler to speedup
float16_inference_program = float_inference_program.clone()
t = fluid.InferenceTranspiler()
t.float16_transpile(float16_inference_program, GPUPlace)

# Running 
float16_results = exe.run(float16_inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)
```

As we can see from the example above, users can simply use the `float16_transpile` method provided by the infernece transpiler class on an existing float inference program to run inference in float16 mode.

### Speedup on GPU
Currently, Fluid inference in float16 mode is only supported on Nvidia GPU device. There is no motivation to support float16 inference on non-ARM CPUs because float16 is not natively supported there and float16 calculation will only be slower than its float counterpart. 

Nvidia started to support its native float16 data type (which has the same internal memory representation as Fluid float16 class) on CUDA 7.5. Moreover, float16 speedups on common computational intensive tasks including GEMM (general matrix-matrix multiplication) and convolution are supported since cublas 7.5 and cuDNN 5.0.

Recently, the introduction of [tensor core](https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/) in volta architecture GPUs and the support of tensor core calculation in CUDA 9.0 and cuDNN 7.0 make float16 truly superior to float in certain deep learning applications. Please refer to this [benchmark report](https://github.com/kexinzhao/Paddle_benchmark/blob/master/float16_benchmark.md) for more details.
