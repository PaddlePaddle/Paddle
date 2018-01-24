# Design Doc: Asynchronous and Non-blocking techniques on heterogeneous devices.


## Problem

We often use heterogeneous devices, like GPU, FPGA, in a deep learning platform. The computation on heterogeneous devices is usually asynchronous and non-blocking. The computation tasks are offloaded to heterogeneous devices and programmers can wait until computation finished and fetched results. Heterogeneous devices can usually execute simultaneously. For example, Fermi architecture (compute capability 2.0+) can simultaneously support

* execute CUDA kernels
* host-to-device memory copy
* device-to-host memory copy.

However, the simultaneous execution is not transparent to programmers. 

Let's use CUDA as an example. There is a building block named `stream` in CUDA. Streams introduce task-based parallelism to CUDA codes. The sequence of operations will be executed in issue-order on the GPU if they are in the same stream. 

The operators in different streams may run concurrently as long as they are in multiple streams and hardware supports it. CUDA Hardware has no notion of streams. The hardware has separate queues (engines) to perform memory copies and to execute kernels.

If we want to take advantage of CUDA devices, we must use at least N streams, where N equals the number of hardware queues. And separate operators into these streams. The N equals to three since CUDA can simultaneously execute CUDA kernels, H2D memcpy, D2H memcpy by the CUDA hardware.

Considering the execution of CUDA devices is asynchronous, there should be a wait operator when switching streams. For example, we want to read the computation result from GPU; then we must wait for the computation complete and issue a device-to-host memory copy.

## Solution

The solution is straightforward based on the hardware properties we described in the problem section. We should:

* Create N device contexts on one device. The N should be corresponding to the hardware property. For example, the CUDA devices should have three device contexts.

* Every tensor should hold the one device context, where the current operator of the tensor is performed on.

* Wait for the execution complete on the previous device context, when switching the current device context of tensors.


The sample C++ program is

```cpp

enum CUDAHardwareStream {
  kCOMPUTATION,
  kD2HMEMCPY,
  kH2DMEMCPY
};

std::map<CUDAHardwareStream, DeviceContext* > gDevCtxs;

class Tensor {
public:
  ...
  
  void SwitchDevCtx(DeviceContext* new_ctx) {
    if (dev_ctx_ != new_ctx) {
        dev_ctx->Wait();
    }
    dev_ctx_ = new_ctx;
  }
  
private:
  ...
  DeviceContext* dev_ctx_;

};


void SomeTensorComputationFunction(Tensor* t) {
  t->SwitchDevCtx(gDevCtxs[kCOMPUTATION]);
  ...
}

```
