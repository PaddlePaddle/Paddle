## Region-based Heterogeneous Memory Management

With the recent explosion of deep learning, entire industries substantial concerns around how to build
a flexible, scalable, reliable deep learning system. Deep learning heavily rely on the capability of mult-GPUs, but GPU memory only the limits available.
A solution to manage the heterogeneous memory resource efficiently is needed.

We provides a technical foundation to reason about challenges and trends.

A memory arena is simply a large, contiguous chunk of memory that is allocated once and then used to manage memory manually by handing out smaller chunks of that memory. You can get memory locations as required from a chunk and return it back to this chunk when you are done with the use.

Instead of requiring individual calls to library malloc/free functions for each small chunk, managing many small chunks of memory this can reduce overhead significantly.

### Abstraction

Unified Interfaces for both CPU and GPU allocation.

```c++
void init();
void shutdown();

void* malloc(majel::Place place, size_t size);
void free(majel::Place place, void* ptr);
size_t memory_used(majel::Place);
```

Each allocator is corresponding to one CPU or GPU device.

1. `init`: calculate and reset how many allocators exist in current system, then respectively recording their own capacity, available, etc.
2. `shutdown`: clear all exist allocators.

3. The definition of `Place` is under folder `paddle/majel`.
4. `memory_used` really makes sense when users do interactive programming, they may wondering the underlying memory resource.

### CPU System Allocators

#### Default Allocator

#### PINNED Allocator

### GPU System Allocators

#### Default Allocator

#### HostFallback Allocator

### Buddy Allocators

### Acknowledge
