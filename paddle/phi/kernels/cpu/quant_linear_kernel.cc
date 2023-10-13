#include "paddle/phi/kernels/impl/quant_linear_kernel_impl.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(
    quant_linear, CPU, ALL_LAYOUT, phi::QuantLinearKernel, float, double) {}
