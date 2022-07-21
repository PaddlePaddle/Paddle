#include "paddle/phi/kernels/average_accumulates_kernel.h"
#include "paddle/phi/kernels/impl/average_accumulates_kernel_impl.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template<>
void GetAccumulators<phi::GPUContext>(
        const phi::GPUContext& dev_ctx,
        const DenseTensor& in_num_accumulates,
        const DenseTensor& in_old_num_accumulates,
        const DenseTensor& in_num_updates,
        int64_t* num_updates,
        int64_t* num_accumulates,
        int64_t* old_num_accumulates) {
    auto stream = dev_ctx.stream();
    auto cuda_place = in_old_num_accumulates.place();
    paddle::memory::Copy(phi::CPUPlace(),
               old_num_accumulates,
               cuda_place,
               in_old_num_accumulates.data<int64_t>(),
               sizeof(int64_t),
               stream);
    paddle::memory::Copy(phi::CPUPlace(),
                num_accumulates,
                cuda_place,
                in_num_accumulates.data<int64_t>(),
                sizeof(int64_t),
                stream);
    paddle::memory::Copy(phi::CPUPlace(),
                num_updates,
                cuda_place,
                in_num_updates.data<int64_t>(),
                sizeof(int64_t),
                stream);
}

template <>
void SetAccumulators<phi::GPUContext>(
        const phi::GPUContext& dev_ctx,
        int64_t num_updates,
        int64_t num_accumulates,
        int64_t old_num_accumulates,                                
        DenseTensor* out_num_accumulates,
        DenseTensor* out_old_num_accumulates,
        DenseTensor* out_num_updates) {
    auto stream = dev_ctx.stream();

    auto cuda_place = out_old_num_accumulates->place();

    paddle::memory::Copy(cuda_place,
                out_old_num_accumulates->data<int64_t>(),
                phi::CPUPlace(),
                &old_num_accumulates,
                sizeof(int64_t),
                stream);
    paddle::memory::Copy(cuda_place,
                out_num_accumulates->data<int64_t>(),
                phi::CPUPlace(),
                &num_accumulates,
                sizeof(int64_t),
                stream);
    paddle::memory::Copy(cuda_place,
                out_num_updates->data<int64_t>(),
                phi::CPUPlace(),
                &num_updates,
                sizeof(int64_t),
                stream);
}

} // namespace phi

PD_REGISTER_KERNEL(average_accumulates,
                   GPU,
                   ALL_LAYOUT,
                   phi::AverageAccumulatesKernel,
                   float,
                   double){}