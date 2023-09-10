
#include "paddle/phi/kernels/impl/fc_kernel_impl.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(fc, GPU, ALL_LAYOUT, phi::FcKernel, float, double) {}