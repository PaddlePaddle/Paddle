# What's difference for fusion kernel?

1. We don't recommend to implement Python API for fusion kernel

  - We don't recommend to implement Python API for fusion kernel, because it contains many inputs or outputs arguments generally, it is difficult to use and understand as an Python API, we recommend to call fusion kernel by pass optimization in dy2static mode or static graph mode.
  - We also don't recommend to reuse fusion kernel in other kernel implementation, but recommended that the fusion kernel be implemented by reusing other kernels.

2. We don't require fusion kernel to have implementations for all devices

  - Fusion Kernel is generally used to accelerate the combined operation on a certain backend. If all backends need to be implemented, the cost is relatively high.
  - We don't recommend implementing a pseudo kernel that just throws exception, if not required, it can be not implemented.
  - If the kernel is only implemented on a certain backend, we recommend add a suffix of backend in kernel name (such as `fused_matmul_onednn`, `fused_fc_xpu`).

3. Fusion Kernel needs to be in the `phi/fusion` namespace.
