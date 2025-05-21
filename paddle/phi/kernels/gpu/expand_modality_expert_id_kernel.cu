#include "paddle/phi/kernels/expand_modality_expert_id_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include <thrust/device_vector.h>

namespace phi {

template<typename T>
void expand_modality_expert_id(const T* expert_id,
                               T* expert_id_out,
                               int64_t seqlen,
                               int64_t k,
                               int64_t num_expert_per_modality,
                               int64_t group_size,
                               int64_t modality_offset,    
                               bool is_group_expert,
                               cudaStream_t stream){
    thrust::transform(
      thrust::cuda::par.on(stream),
      thrust::device_pointer_cast(expert_id),
      thrust::device_pointer_cast(expert_id) + seqlen * k,
      thrust::counting_iterator<T>(0),
      thrust::device_pointer_cast(expert_id_out),
      [k, num_expert_per_modality, group_size, modality_offset, is_group_expert] __device__(T e, T idx) {          
          if (is_group_expert){
            e += idx % k * group_size;
          }
          if (num_expert_per_modality <= 0)
            return static_cast<T>(e);
          T rank = e / num_expert_per_modality;
          T expert_id_in_rank = e % num_expert_per_modality;          
          return static_cast<T>(rank * (num_expert_per_modality * 2)  // HRAD code: only support 2 modality
            + expert_id_in_rank
            + modality_offset * num_expert_per_modality);
      }
    );
}

template <typename T, typename Context>
void ExpandModalityExpertIDKernel(const Context& dev_ctx,
                                  const DenseTensor& expert_id,
                                  int64_t num_expert_per_modality,
                                  int64_t group_size,
                                  int64_t modality_offset,
                                  bool is_group_expert,
                                  DenseTensor* expert_id_out){
  dev_ctx.template Alloc<T>(expert_id_out);
  auto expert_id_shape = expert_id.dims();
  int64_t seqlen = expert_id_shape[0];
  int64_t k = expert_id_shape[1];
  expand_modality_expert_id<T>(expert_id.data<T>(),
                               expert_id_out->data<T>(),
                               seqlen,
                               k,
                               num_expert_per_modality,
                               group_size,
                               modality_offset,
                               is_group_expert,
                               dev_ctx.stream());
}
} // namespace phi

PD_REGISTER_KERNEL(expand_modality_expert_id,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandModalityExpertIDKernel,
                   int,
                   int64_t) {}