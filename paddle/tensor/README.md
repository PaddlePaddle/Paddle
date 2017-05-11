# Tensor: An Unified Data Type in PaddlePaddle

## Pain Point

In this week, we discussed several potential weaknesses of PaddlePaddle caused by rapid iteration and development to promote new business products on the line in recent four years. For instance, current Matrix/Vector implementation in PaddlePaddle are long and tedious to read, which interfered seriously with the contribution of both fresh and professional engineers. More seriously for this issue, it will also become too challenging to maintain over time.


## Learn from Majel

Consequently, we decide to refactor PaddlePaddle step-by-step. First, refactor and replace Matrix/Vector to Tensor, a modern terminology in the deep learning system. Fortunately, we can learn from Majel how to define a Tensor.

To simplify heterogeneous resource allocation in any dimensions (1-9) and types (double, float, float16), Majel consists of several primitives such as `Dim`, `Place` and `Array`, all of them are standard C++ classes.

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

Before writing code, please make sure you already look through Majel Source Code and grabbed the design philosophy of `DArray` in Majel.

To assign subtasks to our colleagues, we have to discuss how to divide it to independent subtasks.

- [ ] 1. First, we need to consider the third-party dependencies in Majel.

    Majel heavily use `boost.variant`, but we don't want to integrate `boost` into PaddlePaddle. It's better to replace boost using the lightweight implementation. https://github.com/mapbox/variant Mapbox variant has the same speedy performance of `boost::variant `but is faster to compile, results in smaller binaries, and has no dependencies.

> @gangliao

- [ ] 2. Re-implement `Place` and `Allocation/Memory`

    I found @wangkuiyi submitted a pull request includes `Place`. @gangliao and @qijun could re-implement `Allocation`, because we have the GPU development experience before joining Paddle team.

> @wangkuiyi @gangliao @qijun

- [ ] 3. Re-implement `Dim`.

    `Dim` is an excellent implementation in Majel. 

> ???

- [ ] 4. Re-implement `Array/Tensor`.

> Prerequisites: 1 - 3

- [ ] 5. Re-implement fundamental operators for `Array/Tensor`.

> Prerequisites: 1 - 4
