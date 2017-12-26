# Region-based Heterogeneous Memory Management
## Design

### Usage

To allocate 4KB CPU memory:

```cpp
p = memory::Alloc(platform::CPUPlace(), 4*1024);
```

To allocate 4KB memory on the 3rd GPU:

```cpp
p = memory::Alloc(platform::CUDAPlace(2), 4*1024);
```

To free memory and check the so-far used amount of memory on a place:

```cpp
auto pl = platform::CUDAPlace(0);
p = memory::Alloc(pl, 4*1024);
cout << memory::Used(pl);
memory::Free(pl, p);
```

### API

In `paddle/memory/memory.h` we have:

```cpp
namespace memory {
template <typename Place> void* Alloc(Place, size_t);
template <typename Place> void Free(Place, void*);
template <typename Place> size_t Used(Place);
}  // namespace memory
```

These function templates have specializations on either `platform::CPUPlace` or `platform::CUDAPlace`:

```cpp
template<>
void* Alloc<CPUPlace>(CPUPlace p, size_t size) {
  return GetCPUBuddyAllocator()->Alloc(size);
}
```

and 

```cpp
template<>
void Alloc<CUDAPlace>(CUDAPlace p, size_t size) {
  return GetGPUBuddyAllocator(p.id)->Alloc(size);
}
```

Similar specializations exist for `Free` and `Used`.

### Implementation

`GetCPUBuddyAllocator` and `GetGPUBuddyAllocator` are singletions.

```cpp
BuddyAllocator* GetCPUBuddyAllocator() {
  static BuddyAllocator* a = NULL;
  if (a == NULL) {
    a = new BuddyAllocator(new CPUAllocator /*backup allocator*/, ...);
  }
  return a;
}

BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
  static BuddyAllocator* as = NULL;
  if (as == NULL) {
    as = new BuddyAllocator*[platform::NumGPUs()];
    for (int gpu = 0; gpu < platform::NumGPUs(); gpu++) {
      as[gpu] = new BuddyAllocator(new GPUAllocator(gpu) /* backup allocator */, ...);
    }
  }
  return as[gpu_id);
```

#### `BuddyAllocator`

`BuddyAllocator` implements the buddy allocation algorithm.  Its constructor takes parameters only related with the algorithm:

```cpp
BuddyAllocator::BuddyAllocator(initial_pool_size, max_pool_size) {
  ...
}
```

Please be aware that **`BuddyAllocator` always allocate aligned memory**, aligned on 32-bytes, which can hold a `BuddyAllocator::Block` object:

```cpp
class BuddyAllocator {
 private:
  struct Block {
    size_t size;
    Block* left, right;
    size_t index; // allocator id
  };
  ...
};
```

Because BuddyAllocator has the meta-data of each block, it can trace the used memory -- record the amount returned by `Alloc` freed in `Free`.  Instead, `CPUAllocator` and `GPUAllocator` doesn't know the size of freed memory block and cannot do the trace.

#### System Allocators

The `GPUAllocator` and `CPUAllocator` are calls *system allocators*.  They work as the fallback allocators of `BuddyAllocator`.

## Justification

I got inspiration from Majel and Caffe2, though above design look different from both.

### Caffe2

In Caffe2, `Tensor<Context>::mutable_data()` allocates the memroy.  In particular, [`Tensor<Context>::mutable_data`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/tensor.h#L523) calls [`Tensor<Context>::raw_mutable_data`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/tensor.h#L459), which in turn calls [`Context::New`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/tensor.h#L479).

There are two implementations of `Context`:

1. [`CPUContext`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context.h#L105), whose [`New` method](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context.h#L131) calls [`g_cpu_allocator.get()->New(size_t)`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context.cc#L15) to allocate the memory.

1. [`CUDAContext`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context_gpu.h#L99), which has a data member [`int gpu_id_`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context_gpu.h#L202).  This looks very similar to class `majel::CUDAPlace`, who also has an `int id_` data member.   `CUDAContext::New(size_t)` calls [`g_cub_allocator->DeviceAllocate(&ptr, nbytes)`](https://github.com/caffe2/caffe2/blob/v0.7.0/caffe2/core/context_gpu.cu#L355) to allocate the memory.

### Majel

In Majel, there are basically two allocator types:

1. `cpu::SystemAllocator`, which has similar functionality to `caffe2::CPUContext::New/Delete`.
1. `gpu::SystemAllocator`, which has similar functionality to `caffe2::CUDAContext::New/Delete`.

However, memory allocation is not via these two allocators.  Instead, these two allocators are defined in hidden namespaces.

In Majel there are hidden global variables like:

1. `cpu::SystemAllocator g_cpu_allocator`, and
1. `vector<gpu::SystemAllocator*> g_gpu_allocators(NUM_GPUS)`.

Programs allocate memory via a BuddyAllocator, which can take the `g_cpu_allocator` or a `g_gpu_allocators[gpu_id]` as its *fallback allocator*, so that if BuddyAllocator cannot find a block in its memory pool, it extends its memory pool by calling the fallback allocator's `New(size_t)`.
