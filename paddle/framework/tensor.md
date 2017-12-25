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
typedef boost::variant<CUDAPlace, CpuPlace> Place;
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

`DDim` plays two kinds of roles in Majel. First, it is used to indicate the size of a tensor. For example, we can construct a new `DArray` by following way:
 
 ```c++
 DArray arr = make_darray(make_ddim({2,3}), 0.0f);
 ```
 It means that `arr` will be a two-dimension tensor, or a matrix. The size of its first dimension is 2 and the second is 3. All the element value of `arr` will be initialized as 0.0 .
 
 The second meaning of `DDim` is tensor index. For example, if we want to access the value in the 1st row and 2nd column of `arr` and set it to 1.0, we can do like this:

 ```c++
 arr[make_ddim({0, 1})] = 1.0；
 ```

## Implement Tensor in Paddle

We want to create a Tensor class to replace Vector and Matrix, and to support high-dimensional data. The operations on Tensor are implemented in both CPU and GPU. We also want to make sure that the Tensor interface is friendly to its callers.

Tensor is only responsible for describing computing. It will not take charge of memory allocation policy, handles of some CUDA library context(e.g. cublasHandle, cudnnHandle), and dispatching CUDA kernels. Paddle has realize the initialization and resources management of hardware.

Before writing code, please make sure you already look through Majel Source Code and grabbed the design philosophy of `DArray` in Majel.


### Memory Management
`Allocation` manages a block of memory in device(CPU/GPU). We use `Place` to decribe memory location. The details of memory allocation and deallocation are implememted in `Allocator` and `DeAllocator`. Related low-level API such as `hl_malloc_device()` and `hl_malloc_host()` are provided by Paddle.

### Dim and Array
#### Dim

`Dim` decribes the dimension information of an array.

`DDimVar` is an alias of a specializd class of boost.variant class template.

`DDim` is introduced to represent a dynamically sized dimension.

For example:

```
Dim<2> d1 = make_dim(3, 3);
DDim d2 = make_ddim({1, 2, 3});
```

You must appoint a concrete sized dimension to Dim, whereas DDim can represent a dynamically sized dimension.
#### Array

`Array` represents for a tensor with specific type and size.

`DArrarVar` is an alias of a specialized class of boost.variant class template.

`DArray` is introduced to represent a dynamically typed array.

For example:

```
Array<float, 2> a1(Dim<2>(2, 2));
DArray a2 = make_darray(make_ddim({3, 4}), 0.0, CpuPlace());
```

You must appoint the type and dimension of a Array, whereas DArray can represent a dynanmically typed array.


Please reference the section of `Learn from Majel` for more details.

### ArrayView

`ViewIterator` is a class template which implements basic iterator operation, including increment(++), decrement(--), dereference(*), equality comparisons(==) and so on.

`ArrayView` is an encapsulation of `Array`， which introduces extra iterator methods, such as `begin()` and `end()`. The `begin()` method returns an iterator pointing to the first element in the ArrayView. And the `end()` method returns an iterator pointing to the pass-the-end element in the ArrayView.

`ArrayView` make the visting and manipulating an array more efficiently, flexibly and safely.


A global function `make_view` is provided to transform an array to corresponding arrayview.

```
template<typename T, int D>
ArrayView<T, D> make_view(const Array<T, D>& in) {
    return in;
}
```

A global function `make_iterator` is provided to make iterator of an array.

```
template<typename T, int D>
ViewIterator<ArrayView<T, D>> make_iterator(const Array<T, D>& in, Dim<D> idx) {
    return make_iterator(make_view(in), idx);
}
```

### Basic Operations

The operations that manipulate DArray are defined as global functions, such as `ones`, `zeros`, `reshape`, `gemm` and so on.

An array will be trasformed into an arrayview and then passed to the operation launching on a specific device(CPU/GPU).
