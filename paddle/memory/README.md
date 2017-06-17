## Region-based Heterogeneous Memory Management

As usual, you can find all the gory details about region-based memory management in [wikipedia](https://en.wikipedia.org/wiki/Region-based_memory_management).
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

3. The definition of `Place` is under folder `paddle/majel`. It denotes allocation places either CPU or GPU.
4. `memory_used` really makes sense when users do interactive programming such as Python, they may wondering the underlying memory resource.

### CPU System Allocators

#### Default Allocator

#### PINNED Allocator

### GPU System Allocators

#### Default Allocator

#### HostFallback Allocator

### Buddy Allocators

### Acknowledge
