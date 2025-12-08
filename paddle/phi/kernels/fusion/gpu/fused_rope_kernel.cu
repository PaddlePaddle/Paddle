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

template <typename T>
__device__ void fused_rope_block_forward(const T *src, const T *sin,
                                         const T *cos,
                                         const int64_t *position_ids, T *dst,
                                         const bool interleaved, const int s_id,
                                         const int b_id, const int offset_block,
                                         const int offset_block_dst,
                                         const int h, const int d,
                                         const int stride_h, const int stride_d,
                                         const int o_stride_h,
                                         const int o_stride_d,
                                         const float rotary_emb_base,
                                         const int seq_len_for_pos) {
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d; i += blockDim.x * blockDim.y) {
    int64_t pos = s_id;
    if (position_ids) {
      pos = position_ids[b_id * seq_len_for_pos + s_id];
    }

    if (sin && cos) {
      shared_mem_sin[i] = static_cast<float>(sin[pos * d + i]);
      shared_mem_cos[i] = static_cast<float>(cos[pos * d + i]);
    } else {
      float idx = (float)((i / 2) * 2);
      float inv_freq = 1.0f / powf(rotary_emb_base, idx / (float)d);
      float freq = (float)pos * inv_freq;
      sincosf(freq, &shared_mem_sin[i], &shared_mem_cos[i]);
    }
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = static_cast<float>(src[offset_src]);
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate =
            (d_id + d / 2 < d)
                ? -static_cast<float>(src[offset_src + (d / 2) * stride_d])
                : static_cast<float>(src[offset_src + (d / 2 - d) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           ? -static_cast<float>(src[offset_src + stride_d])
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = static_cast<T>(v_src * v_cos + v_src_rotate * v_sin);
    }
  }
}

template <typename T>
__global__ void fused_rope_forward_kernel(
    const T *src, const T *sin, const T *cos, const int64_t *position_ids,
    T *dst, const bool interleaved, const int h, const int d,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, const float rotary_emb_base,
    const int seq_len_for_pos) {
  int s_id = blockIdx.x;
  int b_id = blockIdx.y;

  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;

  fused_rope_block_forward(src, sin, cos, position_ids, dst, interleaved, s_id,
                           b_id, offset_block, offset_block_dst, h, d, stride_h,
                           stride_d, o_stride_h, o_stride_d, rotary_emb_base,
                           seq_len_for_pos);
}

template <typename T, typename Context>
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const paddle::optional<DenseTensor>& k,
                     const paddle::optional<DenseTensor>& v,
                     const paddle::optional<DenseTensor>& sin,
                     const paddle::optional<DenseTensor>& cos,
                     const paddle::optional<DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     float rotary_emb_base,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  int64_t numel = q.numel();
  dev_ctx.template Alloc<T>(out_q);
  if (k) dev_ctx.template Alloc<T>(out_k);
  if (v) dev_ctx.template Alloc<T>(out_v);
  if (numel <= 0) return;

  auto batch_size = time_major ? q.dims()[1] : q.dims()[0];
  auto seq_len = time_major ? q.dims()[0] : q.dims()[1];
  auto num_heads = q.dims()[2];
  auto head_dim = q.dims()[3];

  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  auto stream = dev_ctx.stream();
  const T* sin_data = sin.get_ptr() ? sin.get_ptr()->data<T>() : nullptr;
  const T* cos_data = cos.get_ptr() ? cos.get_ptr()->data<T>() : nullptr;
  const int64_t* position_ids_data =
      position_ids.get_ptr() ? position_ids.get_ptr()->data<int64_t>() : nullptr;

  dim3 grid(seq_len, batch_size);
  dim3 block(32, 4); // 32 threads per warp, 4 warps per block
  size_t shared_mem_size = 2 * head_dim * sizeof(float);

  // Q
  int stride_s_q = time_major ? q.strides()[0] : q.strides()[1];
  int stride_b_q = time_major ? q.strides()[1] : q.strides()[0];
  int stride_h_q = q.strides()[2];
  int stride_d_q = q.strides()[3];

  int o_stride_s_q = time_major ? out_q->strides()[0] : out_q->strides()[1];
  int o_stride_b_q = time_major ? out_q->strides()[1] : out_q->strides()[0];
  int o_stride_h_q = out_q->strides()[2];
  int o_stride_d_q = out_q->strides()[3];

  fused_rope_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
      q.data<T>(), sin_data, cos_data, position_ids_data, out_q->data<T>(),
      use_neox_rotary_style, num_heads, head_dim, stride_s_q, stride_b_q,
      stride_h_q, stride_d_q, o_stride_s_q, o_stride_b_q, o_stride_h_q,
      o_stride_d_q, rotary_emb_base, seq_len);

  // K
  if (k) {
    auto k_num_heads = k->dims()[2];
    int stride_s_k = time_major ? k->strides()[0] : k->strides()[1];
    int stride_b_k = time_major ? k->strides()[1] : k->strides()[0];
    int stride_h_k = k->strides()[2];
    int stride_d_k = k->strides()[3];

    int o_stride_s_k = time_major ? out_k->strides()[0] : out_k->strides()[1];
    int o_stride_b_k = time_major ? out_k->strides()[1] : out_k->strides()[0];
    int o_stride_h_k = out_k->strides()[2];
    int o_stride_d_k = out_k->strides()[3];

    fused_rope_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        k->data<T>(), sin_data, cos_data, position_ids_data, out_k->data<T>(),
        use_neox_rotary_style, k_num_heads, head_dim, stride_s_k, stride_b_k,
        stride_h_k, stride_d_k, o_stride_s_k, o_stride_b_k, o_stride_h_k,
        o_stride_d_k, rotary_emb_base, seq_len);
  }

  // V
  if (v) {
    auto v_num_heads = v->dims()[2];
    int stride_s_v = time_major ? v->strides()[0] : v->strides()[1];
    int stride_b_v = time_major ? v->strides()[1] : v->strides()[0];
    int stride_h_v = v->strides()[2];
    int stride_d_v = v->strides()[3];

    int o_stride_s_v = time_major ? out_v->strides()[0] : out_v->strides()[1];
    int o_stride_b_v = time_major ? out_v->strides()[1] : out_v->strides()[0];
    int o_stride_h_v = out_v->strides()[2];
    int o_stride_d_v = out_v->strides()[3];

    fused_rope_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        v->data<T>(), sin_data, cos_data, position_ids_data, out_v->data<T>(),
        use_neox_rotary_style, v_num_heads, head_dim, stride_s_v, stride_b_v,
        stride_h_v, stride_d_v, o_stride_s_v, o_stride_b_v, o_stride_h_v,
        o_stride_d_v, rotary_emb_base, seq_len);
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16){};
