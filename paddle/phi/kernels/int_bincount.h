#include "paddle/phi/core/utils/data_type.h"
#include "paddle/common/flags.h"
#include <vector>
#include <cstdint>
#include "cub/device/device_histogram.cuh"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi{

template<typename T, typename Context>
void IntBincount(const Context& ctx, 
                                    const DenseTensor &x, 
                                    int64_t low, 
                                    int64_t high, 
                                    int64_t out_dtype,
                                    DenseTensor* out);
}