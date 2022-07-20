#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/average_accumulates_kernel.h"
#include "paddle/phi/kernels/impl/average_accumulates_kernel_impl.h"

namespace phi{

template <>
void GetAccumulators<phi::CPUContext>(const phi::CPUContext& dev_ctx,
                                      const DenseTensor& in_num_accumulates,
                                      const DenseTensor& in_old_num_accumulates,
                                      const DenseTensor& in_num_updates,
                                      int64_t* num_updates,
                                      int64_t* num_accumulates,
                                      int64_t* old_num_accumulates) {
  *old_num_accumulates = in_old_num_accumulates.data<int64_t>()[0];
  *num_accumulates = in_num_accumulates.data<int64_t>()[0];
  *num_updates = in_num_updates.data<int64_t>()[0];
}

template <>
void SetAccumulators<phi::CPUContext>(const phi::CPUContext& dev_ctx,
                                      int64_t num_updates,
                                      int64_t num_accumulates,
                                      int64_t old_num_accumulates,
                                      DenseTensor* out_num_accumulates,
                                      DenseTensor* out_old_num_accumulates,
                                      DenseTensor* out_num_updates) {
  out_old_num_accumulates->data<int64_t>()[0] = old_num_accumulates;
  out_num_accumulates->data<int64_t>()[0] = num_accumulates;
  out_num_updates->data<int64_t>()[0] = num_updates;
}

} // namespace phi

PD_REGISTER_KERNEL(
    average_accumulates,
    CPU,
    ALL_LAYOUT,
    phi::AverageAccumulatesKernel,
    float,
    double){}