# What's difference for fusion kernel?

1. We don't recommend to implement Python API for fusion kernel

  - We don't recommend to implement Python API for fusion kernel, because it contains many inputs or outputs arguments generally, it is difficult to use and understand as an Python API, we recommend to call fusion kernel by pass optimization in dy2static mode or static mode.
  - We also don't recommend to reuse fusion kernel in other kernel implementation, but recommended that the fusion kernel be implemented by reusing other kernels.

2. We don't require fusion kernel to have implementations for all devices

  - Fusion Kernel is generally used to accelerate the combined operation on a certain device. If all devices need to be implemented, the cost is relatively high.
  - We don't recommend implementing a pseudo kernel that just throws exception, if not required, it can be not implemented.

3. Fusion Kernel needs to be in the `phi/fusion` namespace.

4. The file naming of the Fusion Kernel needs to follow the format of `fused_[fusion operation name]_kernel.h/cc/cu`, the kernel function naming of the Fusion Kernel needs to follow the format of `Fused[fusion operation name]Kernel`, and the kernel registration naming of the Fusion Kernel needs to follow the format of `fused_[fusion operation name]`.
