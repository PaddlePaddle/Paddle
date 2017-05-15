# Tensor: An Unified Data Type in PaddlePaddle

## Pain Point

In this week, we discussed several potential weaknesses of PaddlePaddle caused by rapid iteration and development to promote new business products on the line in recent four years. For instance, current Matrix/Vector implementation in PaddlePaddle are long and tedious to read, which interfered seriously with the contribution of both fresh and professional engineers. More seriously for this issue, it will also become too challenging to maintain over time.


## Learn from Majel

Consequently, we decide to refactor PaddlePaddle step-by-step. First, refactor and replace Matrix/Vector to Tensor, a modern terminology in the deep learning system. Fortunately, we can learn from Majel how to define a Tensor.

To simplify heterogeneous resource allocation in any dimensions (1-9) and types (double, float, float16), Majel consists of several primitives such as `Dim`, `Place` and `Array`, all of them are standard C++ class templates.

1. `Place`: memory location [i.e. CPU/GPU].
2. `Allocation`: heterogeneous resource allocator [i.e. 20MB in GPU].
3. `Dim`: size of each dimension. [i.e. Dim<4>({10, 2, 5, 1})]
4. `Array`: dynamic array consists of `Place`, `Dim`, and a pointer to memory.

If you dig deeper into Majel source code, you will find Majel heavily use `boost.variant`. The variant class template is a safe, generic, stack-based discriminated union container, **offering a simple solution for manipulating an object from a heterogeneous set of types in a uniform manner**. Whereas standard containers such as std::vector may be thought of as "multi-value, single type," variant is "multi-type, single value."

As a simple example, consider the following:

```c++
#include "boost/variant.hpp"
#include <iostream>

class my_visitor : public boost::static_visitor<int>
{
public:
    int operator()(int i) const
    {
        return i;
    }
    
    int operator()(const std::string & str) const
    {
        return str.length();
    }
};

int main()
{
    boost::variant< int, std::string > u("hello world");
    std::cout << u; // output: hello world

    int result = boost::apply_visitor( my_visitor(), u );
    std::cout << result; // output: 11 (i.e., length of "hello world")
}
```

In Majel, `DDimVar` is derived from `Dim`, `DArrayVar` is from `Array`.

```c++
template<int i>
struct Dim {
...    
int head;
Dim<i-1> tail;
}
```

```c++
template<typename T, int D>
class Array : public Buffer {
    ...
private:
    Dim<D> size_;
    Dim<D> stride_;
    T* ptr_;
};
```

```c++
typedef boost::variant<GpuPlace, CpuPlace> Place;
typedef boost::variant<Dim<1>, Dim<2>, Dim<3>, Dim<4>, Dim<5>,
                       Dim<6>, Dim<7>, Dim<8>, Dim<9>> DDimVar;
typedef boost::variant<
    Array<float, 1>,
    Array<float, 2>,
    Array<float, 3>,
    Array<float, 4>,

    Array<double, 1>,
    Array<double, 2>,
    Array<double, 3>,
    Array<double, 4>,

    Array<float16, 1>,
    Array<float16, 2>,
    Array<float16, 3>,
    Array<float16, 4> > DArrayVar;
```

Because `variant` may be thought of as "multi-type, single value", we can utilize it to implement unified interfaces for PaddlePaddle.

## implement Tensor in Paddle

We aim to implement an independent tensor library, which provides flexiable and efficient tensor operations and can run on muti-devices(CPU and GPU). In brief, there are mainly two aspects to consider. First, we have to make encapsulation of the hardware for the tensor library, including device memory, computing handles, cuda streams and so on. Second, after the tensor libaray is able to access the hardware resources, we should make carefully design and abstraction of the upper levels and provide flexiable interfaces.  

Before writing code, please make sure you already look through Majel Source Code and grabbed the design philosophy of `DArray` in Majel.

### Resources Manegement

#### Device Initializer


At first, we locate which context runs the tensor library, and then get the GPU device infomation.
 
```
int count = get_cuda_device_count();
for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    deviceProperties[i] = prop;
}
```

Second, the system memory allocator is created.

```
gpu::SystemAllocator::init();
cpu::SystemAllocator::init();
detail::SharedAllocator::init();

```


Third, we make initalization in each GPU.

```
void init_device(int i) {
    detail::rand_init(GpuPlace(i));
    gpu::detail::cublas_init(GpuPlace(i));
#ifdef USE_CUDNN
    gpu::detail::cudnn_init(GpuPlace(i));
#endif
}
```

```
int gpu_count = majel::gpu::detail::get_device_count();
for (int i = 0; i < gpu_count; ++i) {
     init_device(i);
}
``` 

The cublasHandle and cudnnHandle will be created for each GPU card.


#### Memory Allocator
Majel has a malloc module to manage the device memory, including initialization, allocation and deallocation. We could reference the elegent design of majel.

TBD.

#### Stream Scheduler

A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code. While operations within a stream are guaranteed to execute in the prescribed order, operations in different streams can be interleaved and, when possible, they can even run concurrently.

Majel have a module named scheduler to schedule cuda kernel operation.

At first, `GpuLaunchInfo` is defined as following.

```
struct GpuLaunchInfo {
    int            device;
    cudaStream_t   stream;
};
```

Second, a cuda kernel will be set into a specific stream in launch info waiting for executing. There are two examples below.

1.Convlution operation with cuDNN.

```
cudnnHandle_t prepare_launch(scheduler::GpuLaunchInfo launch) {
    cudnnHandle_t handle = majel::gpu::detail::cudnn_state[launch.device];

    majel::gpu::detail::set_device(launch.device);
    majel::detail::throw_on_error(cudnnSetStream(handle,
                                                 launch.stream),
                                  "Error when setting stream");
    return handle;
}
```

2.Gemm operation with cuBLAS.

```
template<typename RealType>
void gemm_impl(const Array<RealType, 2> in1, bool in1_T, const Array<RealType, 2> in2,
          bool in2_T, double alpha, Array<RealType, 2> out, double beta, NativeFloatType type) {
    ...

    auto metrics = check_gemm_params(in1, in1_T, in2, in2_T, out);

    majel::detail::GemmVisitor<RealType> mm(in1, in1_T, in2, in2_T,
      alpha, out, beta);
    scheduler::schedule([=](const scheduler::LaunchInfo& li) {
          boost::apply_visitor(mm, li);
      }, {in1, in2, out}, {out}, std::move(metrics));
}
```

The stream will be created in `GpuEvent` in scheduler module.

```
GpuEvent::GpuEvent(int device, int priority)
: device_(device), record_timing_(false) {

    gpu::detail::set_device(device);

    stream_      = gpu::detail::create_new_stream_with_priority(priority);
    start_event_ = nullptr;
    end_event_   = nullptr;
}
```

Majel has a detailed and powerful scheduler, whereas paddle creates some streams by default in initialization stage and make some manaul scheduling policy. 

Where should we implement the scheduler module, in or out the tensor library? 




### Abstract Concepts

#### DArray and the related

Please refer to [Learn from majel](#Learn from majel).

1. `Place`: memory location [i.e. CPU/GPU].
2. `Allocation`: heterogeneous resource allocator [i.e. 20MB in GPU].
3. `Dim`: size of each dimension. [i.e. Dim<4>({10, 2, 5, 1})]
4. `Array`: dynamic array consists of `Place`, `Dim`, and a pointer to memory.


#### Matrix Operation
Majel provides lots of matrix operation for users.

TBD.


#### Lazy Operation

Lazy operation can avoid some temporary memory allocation and merge kernels.
Majel support lazy operation. And Paddle also implement lazy operation using expression templates.

TBD.

#### Sparse Matrix

We can represent Sparse Matrix using a conbination of DArrayVar.

```
struct SparseMatrix {
    DArrayVar row_;
    DArrayVar col_;
    DArrayVar value_;
    
    size_t nnz_;
}
```

TBD.
