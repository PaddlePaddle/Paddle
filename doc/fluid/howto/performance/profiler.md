## Introduction

There are many performance analysis tools for [different programming languages and different software frameworks](https://en.wikipedia.org/wiki/List_of_performance_analysis_tools). For most popular deep learning frameworks, they use several programming languages and adapt to heterogeneous platforms. Similar to most of the deep learning frameworks, PaddlePaddle also uses C++, CUDA and Python as the basic programming languages to adapt to run on CPU and GPU devices.  The [`nvprof` tools](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview) is usually used to analyse the CUDA program.  We have [a document](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/optimization/cpu_profiling.md) to profile CPU and Python program by [yep](https://pypi.python.org/pypi/yep) and [Google's perftools](https://github.com/google/pprof) to profile only the CPU and Python program. But for [PaddlePaddle fluid](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/fluid.md), the operator is the basic computing unit. The developers usually want to collect the time of each operator and locate bottlenecks.  The `nvprof` usually collect the timeline of CUDA-related activities on both CPU and GPU, including kernel execution, memory transfers, memory set and CUDA API calls and events or metrics for CUDA kernels. And the `yep` and `Google's perftools` can't collect the timeline for CUDA program. All these tools can't collect time in the operator level. So we design this profiling tool.

## Architecture

The work flow for most task is as follows. Each operator will run many times in the all iterations. So the profiler must collect the total time of each operator during the iteration. For more, sometimes, the developers may want to collect more detailed time span inside the operator or record time span for elsewhere, this requires that the profiler must support to record the nested time span. And in order to speedup training, all the deep learning frameworks support parallel computing, including multiple threads on CPU and multiple GPUs. So the profiler must be able to collect the timeline for each thread. In addition, the profiler also occupies certain resources. It must can be easily to be enabled or disabled by the developers. At last, the profiler should present a human-readable report.  

```python
for i in xrange(M):  # M is  the iteration number
  for op in operator_lists: # The `operator_lists` contains all the operators in the network.
    op.run();
```

In summary, the proflier should have following features:

- records time span in loop.
- supports nested time span.
- supports multiple threads/multiple GPUs.
- supports to be enabled and disabled by users.

But how to record the time for the mixed C++ and CUDA program?  There many C++ APIs to get the current calendar time in host program. But for GPU, the CUDA kernels may be executed concurrently if they are in different [streams](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) and the CUDA kernels is asynchronous with the host program if there is no the synchronous aftern the CUDA kernels. CUDA provides [event](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events) to monitor the device and perform accurate timing. Inspired by PyTorch and CUDA event, we also design and apply the events to record the timeline. Then summarize and present statistics based on these events.  

The overall flow is shown as the following figure.

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/profiler.png" align="center"/><br/>

### Event

In above work flow, a pair of events are needed before and after the piece of code to collect time. So the event has a flag to mark whether it is a starting event or an ending event. Except this two kinds of event, sometime, a only marker with a text message is needed, for example, a marker to specify the profiling start or end. There are three kinds of event:

```c++
enum EventKind {
  kMark,
  kPushRange,
  kPopRange};
```
- kMark: only a marker without time range.
- kPushRange: mark the starting event for time range.
- kPopRange: mark the ending event for time range.

For the CPU code, the events only need to record the current time. For the CUDA code, the [event management functions of CUDA](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT) are used.  For many pieces of code, an event lists are used to record each piece.

```c++
class Event {
 public:
  // The DeviceContext is used to get current  CUDA stream.
  Event(EventKind kind, std::string name, uint32_t thread_id,
        const platform::DeviceContext* dev_ctx = nullptr);
  double CpuElapsedUs(const Event& e) const;
  double CudaElapsedUs(const Event& e) const;

 private:
  EventKind kind_;
  std::string name_;
  uint32_t thread_id_;
  int64_t cpu_ns_;
#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event_ = nullptr;
  int device_ = -1;
#endif
};

struct EventList {
  std::forward_list<std::vector<Event>> event_blocks;
};
```

As mentioned above, there is no need to record the timeline when disabling the profiler. So there is a global state to enable or disable the profiler.

```c++
enum ProfilerState {
  kDisabled,
  kCPU,
  kCUDA
};
ProfilerState g_state;
```
- kDisabled: the disabled state.
- kCPU: CPU profiling state.
- kCUDA: GPU profiling state.

A pair of starting and ending events are pushed to event lists in constructor and destructor of `RecordEvent`. So the timeline is recorded for the code in the lifecycle of an object of `RecordEvent`.

```c++
struct RecordEvent {
  explicit RecordEvent(const std::string name,
                       platform::DeviceContext* dev_ctx = nullptr) {
    if (kState == ProfilerState::kDisabled) return;
    // push the starting event to the event lists.
  }
  ~RecordEvent() {
    if (kState == ProfilerState::kDisabled) return;
    // push the ending event to the event lists.
  }
};
```
