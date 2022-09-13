![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Programming Guidelines")

[README](/README.md#documentation) > **Programming Guidelines**

# Programming Guidelines

## Hierarchical Organization

CUTLASS embodies a design paradigm exemplified by the [CUB library](https://nvlabs.github.io/cub/) 
for expressing collective operations. Objects expose an interface for a problem that is then decomposed 
into concurrent subtasks executed by cooperating threadblocks, warps, and threads. For example, a grid-level 
object may be constructed with base pointers to the start of a GEMM operation, add a threadblock-dependent 
offset to partition the problem, and then compute a per-threadblock GEMM. This in turn performs some 
operations as a collection of cooperating threads, while it may partition other parts of the task into 
warp-level subtasks.

Consequently, CUTLASS components are organized by the computation then by the layer of
the following hierarchy.

* *device*: an operation is _device-wide_ and may launch one or more kernels on the GPU
* *kernel*: an operation is implemented by a CUDA kernel with definitions for `__shared__` memory and constant memory allocations
* *threadblock*: an operation is collectivey executed by a threadblock; any component calling `__syncthreads()` is likely to be threadblock-scope
* *warp*: an operation is collectively executed by a warp; threads within the context of a warp are referred to as _lane_
* *thread*: an operation is performed by an individual thread with no other data sharing or interaction with other threads
* *instruction*: an operation corresponds to an individual hardware or PTX instruction

## Design Patterns

CUTLASS strives to achieve the highest performance possible on NVIDIA GPUs while also offering a
flexible composition that an be easily applied to solve new problems related to Deep Learning and
linear algebra. Though we intend to make CUTLASS as simple and straightforward as possible, given
a tradeoff between simplicity and performance, CUTLASS chooses performance. Consequently, several
design patterns are necessary to yield a composable structure while also satisfying these performance
objectives. This section is intended to provide more detail.

### Templates

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:
* Mixed precision arithmetic and data storage
* Kernels specialized for layout and problem size
* Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.

### Constant Memory

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel 
launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several 
offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements 
of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists 
only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing 
a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant 
memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class 
which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params` 
structure should also include a data member corresponding to each data member in the parent class, so these too can 
be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as 
its first argument.


### Composable Shared Memory

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm 
introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held 
in shared memory. Any object requiring shared memory storage for itself or its data members should define a child 
structure called `SharedStorage`. This holds data needed by the class and also instantiates `SharedStorage` 
objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements. 
Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes 
of child objects are known to be non-overlapping, unions may be used to alias multiple SharedStorage objects to the same
shared memory region and reduce overall SMEM capacity.

### Loop Unrolling

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

All loops expected to be unrolled should be annotated with `CUTLASS_PRAGMA_UNROLL` to explicitly direct the compiler
to unroll them. 

```c++
int const kN = 8;
Array<float, kN> x;                       // Array we would like to store in registers

CUTLASS_PRAGMA_UNROLL                     // Directs the CUDA compiler to unroll this loop.
for (int idx = 0; idx < kN; ++idx) {      // Loop has constant number of iterations.

  x[i] = float(idx);                      // Indirect access by induction variable results in 
                                          // direct register access.
}
```

## Style

### C++ Style

CUTLASS source code follows the 
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with exceptions and extensions.

Design choices should be consistent with the 
[CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md) recommendations by Stroustrup and Sutter.

### CUDA Built-in Variables

Avoid direct access to CUDA built-in variables `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` within
CUTLASS components except in special circumstances. 

Using built-in 'global' variables directly within resuable components necessitates that all components
use them consistently which may not be possible if CUTLASS components are used in other contexts.

Instead, components should accept a linear ID identifying threads, warps, and threadblocks from calling
code. The top-level kernel may then decide how to map threads, warps, and blocks to the problem it is
solving.

### Use CUTLASS Fundamental Types

Use the [fundamental types](fundamental_types.md) defined in CUTLASS consistently. Doing so contributes
to a framework of interoperable, consistent components.

In particular, be sure to use:

* [Numeric types](fundamental_types.md#numeric-types) to represent numeric data in host and device code
* [Containers](fundamental_types.md#containers) to store data in register-backed arrays
* [functional.h](fundamental_types.md#functional) to perform numeric operations in generic code
* [Layouts](layout.md) to store stride and partially specialize template classes
* [`TensorRef` and `TensorView`](layout.md#tensorref) to pass pointers and layout objects

Avoid defining alternative implementations of the same functionality. Instead, prefer to enhance
or extend additional components where it makes sense.

### Classes and Structs

Type names use `CapitalLetters` except when implementations are a _perfect_ drop-in replacement for
Standard Library components.

Follow the [CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-struct) 
to decide whether to use `class` or `struct`. Namely,
* use `class` when the object must maintain an invariant. Data members related to the invariant should be private.
* use `struct` when the class has no invariant to maintain, and data members may vary arbitrarily.

### Class Members

Methods and members are written using `snake_case`.

Private data and function members have suffix `_`.

### Constant names

CUTLASS makes extensive use of constants and compile-time evaluation. Constant variable names should have
prefix `k` and use mixed case. True compile-time constsants should be defined as `constexpr` to enable
dependent `constexpr` functions.

CUTLASS uses ["East const"](http://slashslash.info/2018/02/a-foolish-consistency/) style, placing `constexpr` keyword
after the type name.

```c++
float constexpr kPi = 3.14159f;
```

### Class Member Order

Members within classes and structures should be organized as follows:

1. Type and constant definitions
2. Data members
3. Constructors
4. Other methods

This convention follows the [CUB library](https://nvlabs.github.io/cub/) and is also described by 
[Howard Hinnant](https://howardhinnant.github.io/classdecl.html). Unsurprisingly, it approximates 
the usual ordering of chapters in a typical Systems and Controls textbook. That is,
(1.) identify relevant constants, (2.) define a state-space representation of the dynamical system 
under study (i.e. the data members), and (3.) devote subsequent chapters to definining dynamical behavior
of the system (i.e. the methods).

_Example_:
```c++
class A {
public:
  // Type definitions
protected:
  // protected Type definitions
private:
  // private Type definitions

public:
  // Data members
protected:
  // protected data members
private:
  // private data members

public:
  // Methods
protected:
  // protected methods
private:
  // private methods

};

```

### File Names

Files should be named using `snake_case` with extension `.h` for header files, `.cu` for CUDA sources,
and `.cpp` for C++ host-only source files.

### Use scoped enums

Use scoped enums added in C++11 for enumerated types. Use capital letters for the enumerated type name
and prefix `k` for enumerators like other constants.

```c++
enum class MatrixOperation {
  kNone,
  kTranspose,
  kConjugate,
  kHermitian
};
```

### Namespaces

Namespaces are all lower case. The top-level namespace is `cutlass::`. The second nested namespace refers
top the general category of operation performed by its members, and the third nested namespace refers to
the CUDA execution model scope (if applicable).

The bodies of namespace definitions should not be intented, and comments on the closing brace are welcome.

```c++
namespace cutlass {
namespace gemm {
namespace warp {

struct MmaTensorCore {

};

} // namespace warp
} // namespace gemm
} // namespace cutlass
```

### Macros

Avoid defining macros except where preprocessing is obligatory. In particular, 
avoid using macros for constants.

Several existing macros defined in `cutlass/cutlass.h` are useful for working around compiler-dependent
behavior.

Annotations for device code:
* `CUTLASS_HOST_DEVICE` for functions running on the host and the device
* `CUTLASS_DEVICE` for functions running on the device only

Loop unrolling:
* `CUTLASS_PRAGMA_UNROLL` for full unrolling of loops with constant trip counts
* `CUTLASS_PRAGMA_NO_UNROLL` to prevent unrolling

### #pragma once

Use `#pragma once` to guard all headers.

```c++
/*!

*/

#pragma once

...
```

### Source Line Length

Avoid lines longer than 100 characters. These typically wrap unfavorably when viewed in
Github's pretty printer.


# Copyright

Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
