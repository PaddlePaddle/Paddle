#include "paddle/phi/kernels/moe_gate_dispatch_permute_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h" // 注册相关
#include "paddle/phi/backends/gpu/gpu_context.h" // context相关
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
namespace phi{

template <typename T>
void apply_moe_dispatch_bwd(
    const T* y_grad,
    const float* combine_weights, // [s, k]
    const int* scatter_index,   // [s, k]
    const float* combine_weights_grad,
    const int* expert_id, // [s, k]
    float* gate_logtis_grad,
    T* x_grad,
    int64_t num_rows,
    int64_t k,
    int64_t dim,
    int64_t num_experts,
    int64_t capacity,
    bool use_all2all_permute,
    int64_t world_size,
    int64_t num_local_experts,
    cudaStream_t stream){
    gather_with_mask_launcher<T>(y_grad,
                                scatter_index, 
                                combine_weights,
                                x_grad, num_rows, k, dim, -1, stream, use_all2all_permute, world_size, num_local_experts, capacity);

    topk_grad_with_mask_launcher<float>(combine_weights_grad,
                                        expert_id,
                                        combine_weights,
                                        gate_logtis_grad,
                                        num_rows, k, num_experts, stream);
}


template <typename T, typename Context>
void moe_dispatch_bwd(const Context& dev_ctx,
                      const DenseTensor& combine_weights, // [s, k]
                      const DenseTensor& scatter_index,   // [k, s]
                      const DenseTensor& expert_id, // [s, k]
                      const DenseTensor& y_grad, // [num_experts * capacity, h]
                      const DenseTensor& combine_weights_grad, // [s, k]
                      const DenseTensor&x_grad, 
                      const DenseTensor& gate_logtis_grad,
                      int64_t capacity,
                      bool use_all2all_permute = false,
                      int64_t world_size = -1,
                      int64_t num_local_experts = -1){
  auto combine_weights_dims = combine_weights.dims();
  int64_t num_rows = combine_weights_dims[0];
  int64_t k = combine_weights_dims[1];
  auto y_grad_dims = y_grad.dims();
#ifdef MOE_OPS_AUTO
  int64_t hidden_size = y_grad_dims[2];
#else
  int64_t hidden_size = y_grad_dims[y_grad_dims.size() - 1];
#endif
  int64_t num_experts = gate_logtis_grad.dims()[1];

  apply_moe_dispatch_bwd<T>(
      y_grad.data<T>(),
      combine_weights.data<float>(),
      scatter_index.data<int>(),
      combine_weights_grad.data<float>(),
      expert_id.data<int>(),
      gate_logtis_grad.data<float>(),
      x_grad.data<T>(),
      num_rows,
      k,
      hidden_size,
      num_experts,
      capacity,
      use_all2all_permute,
      world_size,
      num_local_experts,
      dev_ctx.stream());
}

template <typename T, typename Context>
void MoeGateDispatchGradKernel(const Context& dev_ctx,
                               const DenseTensor& combine_weights, // [s, k]
                               const DenseTensor& scatter_index, // [k, s]
                               const DenseTensor& expert_id, // [num_local_experts, num_experts * capacity // num_local_experts, h]
                               const DenseTensor& y_grad, // [s, k]
                               const DenseTensor& combine_weights_grad,
                               int64_t k,
                               int64_t capacity,
                               int64_t world_size,
                               DenseTensor* x_grad,
                               DenseTensor* gate_logtis_grad){
  int64_t num_local_experts = y_grad.dims()[0];
  auto scatter_index_dims = scatter_index.dims();
  DenseTensor t_scatter_index = phi::Empty<T, Context>(dev_ctx, {scatter_index_dims[1], scatter_index_dims[0]});
  phi::Transpose<T, Context>(dev_ctx, scatter_index, {1,0}, &t_scatter_index);
  fleety_utils::TensorTrans2Contiguous(&t_scatter_index);
  moe_dispatch_bwd<T, Context>(dev_ctx,
                               combine_weights,
                               t_scatter_index,
                               expert_id,
                               y_grad,
                               combine_weights_grad,
                               x_grad,
                               gate_logtis_grad,
                               capacity,
                               true, /*use_all2all_permute*/
                               world_size,
                               num_local_experts);

}
} // namespace phi
