## MZONE: Region-based Heterogeneous Memory Management

With the recent explosion of deep learning, entire industries substantial concerns around how to build
a flexible, scalable, reliable deep learning system. Deep learning heavily rely on the capability of mult-GPUs, but GPU memory only the limits available.
A solution to manage the heterogeneous memory resource efficiently is needed.

We provides a technical foundation `MZONE`: region-based heterogeneous memory management to reason about challenges and trends.

A memory arena is simply a large, contiguous chunk of memory that is allocated once and then used to manage memory manually by handing out smaller chunks of that memory. You can get memory locations as required from a chunk and return it back to this chunk when you are done with the use.

### Primary Goal

### Abstraction

### CPU System Allocators

#### Default Allocator

#### PINNED Allocator

### GPU System Allocators

#### Default Allocator

#### HostFallback Allocator

### Buddy Allocators

### Acknowledge
