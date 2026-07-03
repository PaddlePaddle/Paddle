// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/fused_rope_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedRopeGradKernel(const Context& dev_ctx,
                         const optional<DenseTensor>& sin,
                         const optional<DenseTensor>& cos,
                         const optional<DenseTensor>& position_ids,
                         const DenseTensor& dout_q,
                         const optional<DenseTensor>& dout_k,
                         const optional<DenseTensor>& dout_v,
                         bool use_neox_rotary_style,
                         bool time_major,
                         float rotary_emb_base,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  int64_t numel = dout_q.numel();
  dev_ctx.template Alloc<T>(dq);
  if (dout_k) dev_ctx.template Alloc<T>(dk);
  if (dout_v) dev_ctx.template Alloc<T>(dv);
  if (numel <= 0) return;

  auto batch_size = time_major ? dout_q.dims()[1] : dout_q.dims()[0];
  auto seq_len = time_major ? dout_q.dims()[0] : dout_q.dims()[1];
  auto num_heads = dout_q.dims()[2];
  auto head_dim = dout_q.dims()[3];
  auto freqs_head_dim = head_dim;
  PADDLE_ENFORCE_NE(head_dim % 2,
                    1,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  auto stream = dev_ctx.stream();
  const T* sin_data = sin.get_ptr() ? sin.get_ptr()->data<T>() : nullptr;
  const T* cos_data = cos.get_ptr() ? cos.get_ptr()->data<T>() : nullptr;
  const int64_t* position_ids_data =
      position_ids.get_ptr() ? position_ids.get_ptr()->data<int64_t>()
                             : nullptr;

  bool flag_sin_cos = (sin_data && cos_data);
  if (flag_sin_cos) {
    auto sin_dims = sin.get_ptr()->dims();
    freqs_head_dim = sin_dims[sin_dims.size() - 1];
  }

  const int64_t warps_per_block = std::min(num_heads, static_cast<int64_t>(8));
  PADDLE_ENFORCE_LE_UINT32_MAX(seq_len, "fused_rope_grad grid.x");
  PADDLE_ENFORCE_LE_UINT32_MAX(batch_size, "fused_rope_grad grid.y");
  PADDLE_ENFORCE_LE_UINT32_MAX(warps_per_block, "fused_rope_grad block.y");
  dim3 grid(static_cast<uint32_t>(seq_len), static_cast<uint32_t>(batch_size));
  dim3 block(32,
             static_cast<uint32_t>(warps_per_block));  // 32 threads per warp
  size_t shared_mem_size = 2 * head_dim * sizeof(float);

  // Q
  int64_t stride_s_q = time_major ? dout_q.strides()[0] : dout_q.strides()[1];
  int64_t stride_b_q = time_major ? dout_q.strides()[1] : dout_q.strides()[0];
  int64_t stride_h_q = dout_q.strides()[2];
  int64_t stride_d_q = dout_q.strides()[3];

  int64_t o_stride_s_q = time_major ? dq->strides()[0] : dq->strides()[1];
  int64_t o_stride_b_q = time_major ? dq->strides()[1] : dq->strides()[0];
  int64_t o_stride_h_q = dq->strides()[2];
  int64_t o_stride_d_q = dq->strides()[3];

  FusedRopeKernelLauncher(dout_q.data<T>(),
                          sin_data,
                          cos_data,
                          dq->data<T>(),
                          FusedRopeGradKernelImpl<T, int>,
                          FusedRopeGradKernelImpl<T, int64_t>,
                          position_ids_data,
                          flag_sin_cos,
                          use_neox_rotary_style,
                          num_heads,
                          head_dim,
                          freqs_head_dim,
                          stride_s_q,
                          stride_b_q,
                          stride_h_q,
                          stride_d_q,
                          o_stride_s_q,
                          o_stride_b_q,
                          o_stride_h_q,
                          o_stride_d_q,
                          rotary_emb_base,
                          seq_len,
                          batch_size,
                          numel,
                          stream);

  // K
  if (dk && dk->numel() > 0) {
    auto k_num_heads = dk->dims()[2];
    int64_t stride_s_k =
        time_major ? dout_k->strides()[0] : dout_k->strides()[1];
    int64_t stride_b_k =
        time_major ? dout_k->strides()[1] : dout_k->strides()[0];
    int64_t stride_h_k = dout_k->strides()[2];
    int64_t stride_d_k = dout_k->strides()[3];

    int64_t o_stride_s_k = time_major ? dk->strides()[0] : dk->strides()[1];
    int64_t o_stride_b_k = time_major ? dk->strides()[1] : dk->strides()[0];
    int64_t o_stride_h_k = dk->strides()[2];
    int64_t o_stride_d_k = dk->strides()[3];

    FusedRopeKernelLauncher(dout_k->data<T>(),
                            sin_data,
                            cos_data,
                            dk->data<T>(),
                            FusedRopeGradKernelImpl<T, int>,
                            FusedRopeGradKernelImpl<T, int64_t>,
                            position_ids_data,
                            flag_sin_cos,
                            use_neox_rotary_style,
                            k_num_heads,
                            head_dim,
                            freqs_head_dim,
                            stride_s_k,
                            stride_b_k,
                            stride_h_k,
                            stride_d_k,
                            o_stride_s_k,
                            o_stride_b_k,
                            o_stride_h_k,
                            o_stride_d_k,
                            rotary_emb_base,
                            seq_len,
                            batch_size,
                            numel,
                            stream);
  }

  // V
  if (dv && dv->numel() > 0) {
    auto v_num_heads = dv->dims()[2];
    int64_t stride_s_v =
        time_major ? dout_v->strides()[0] : dout_v->strides()[1];
    int64_t stride_b_v =
        time_major ? dout_v->strides()[1] : dout_v->strides()[0];
    int64_t stride_h_v = dout_v->strides()[2];
    int64_t stride_d_v = dout_v->strides()[3];

    int64_t o_stride_s_v = time_major ? dv->strides()[0] : dv->strides()[1];
    int64_t o_stride_b_v = time_major ? dv->strides()[1] : dv->strides()[0];
    int64_t o_stride_h_v = dv->strides()[2];
    int64_t o_stride_d_v = dv->strides()[3];

    FusedRopeKernelLauncher(dout_v->data<T>(),
                            sin_data,
                            cos_data,
                            dv->data<T>(),
                            FusedRopeGradKernelImpl<T, int>,
                            FusedRopeGradKernelImpl<T, int64_t>,
                            position_ids_data,
                            flag_sin_cos,
                            use_neox_rotary_style,
                            v_num_heads,
                            head_dim,
                            freqs_head_dim,
                            stride_s_v,
                            stride_b_v,
                            stride_h_v,
                            stride_d_v,
                            o_stride_s_v,
                            o_stride_b_v,
                            o_stride_h_v,
                            o_stride_d_v,
                            rotary_emb_base,
                            seq_len,
                            batch_size,
                            numel,
                            stream);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16){};
