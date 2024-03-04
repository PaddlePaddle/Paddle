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
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const paddle::optional<DenseTensor>& k,
                     const paddle::optional<DenseTensor>& v,
                     const paddle::optional<DenseTensor>& sin,
                     const paddle::optional<DenseTensor>& cos,
                     const paddle::optional<DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  int64_t numel = q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(out_q);

  phi::Array<int64_t, 3> inputs_num_heads;

  // q.shape: [seq_len, batch_size, num_heads, head_dim] if time_major else
  // [batch_size, seq_len, num_heads, head_dim]

  const int kBatchDimIndex = time_major ? 1 : 0;
  const int kSeqlenDimIndex = time_major ? 0 : 1;

  auto batch_size = q.dims()[kBatchDimIndex];
  auto seq_len = q.dims()[kSeqlenDimIndex];
  inputs_num_heads[0] = q.dims()[2];
  auto head_dim = q.dims()[3];

  int64_t batch_stride_q = q.strides()[kBatchDimIndex];
  int64_t seq_stride_q = q.strides()[kSeqlenDimIndex];
  int64_t batch_stride_kv = batch_stride_q;
  int64_t seq_stride_kv = seq_stride_q;

  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  constexpr const int vec_size = 2;

  auto stream = dev_ctx.stream();

  phi::Array<T*, 3> outs_data;
  phi::Array<const T*, 3> ins_data;
  phi::Array<const T*, 2> sin_cos_data;
  const int64_t* position_ids_data = NULL;

  ins_data[0] = q.data<T>();
  outs_data[0] = out_q->data<T>();
  int num_inputs = 1;

  if (k) {
    dev_ctx.template Alloc<T>(out_k);
    ins_data[num_inputs] = k->data<T>();
    outs_data[num_inputs] = out_k->data<T>();
    inputs_num_heads[num_inputs] = k->dims()[2];

    batch_stride_kv = k->strides()[kBatchDimIndex];
    seq_stride_kv = k->strides()[kSeqlenDimIndex];

    num_inputs++;
  }

  if (v) {
    dev_ctx.template Alloc<T>(out_v);
    ins_data[num_inputs] = v->data<T>();
    outs_data[num_inputs] = out_v->data<T>();
    inputs_num_heads[num_inputs] = v->dims()[2];

    batch_stride_kv = v->strides()[kBatchDimIndex];
    seq_stride_kv = v->strides()[kSeqlenDimIndex];

    num_inputs++;
  }

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType div_c = static_cast<MPType>(1.0f / head_dim);

  bool flag_sin_cos = false;
  if (sin.get_ptr() && cos.get_ptr()) {
    PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                      cos.get_ptr()->dims(),
                      phi::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin.get_ptr()->dims(),
                          cos.get_ptr()->dims()));

    auto sin_dims = sin.get_ptr()->dims();
    int dims_size = sin_dims.size();
    PADDLE_ENFORCE_EQ(
        (dims_size == 2 || dims_size == 4),
        true,
        phi::errors::InvalidArgument("The dims of sin and cos is expected to "
                                     "be 2 or 4, but received %d.",
                                     dims_size));
    if (dims_size == 4) {
      // sin.shape: [1, seq_len, 1, head_dim]
      PADDLE_ENFORCE_EQ(
          (sin_dims[0] == 1 && sin_dims[2] == 1),
          true,
          phi::errors::InvalidArgument(
              "The batch_size and num_heads of sin and cos must be 1."));
    }
    int sin_seq_len_dim = (dims_size) == 4 ? 1 : 0;

    if (position_ids) {
      PADDLE_ENFORCE_EQ(
          (sin_dims[dims_size - 1] == head_dim &&
           sin_dims[sin_seq_len_dim] >= seq_len),
          true,
          phi::errors::InvalidArgument(
              "The seq_len of sin and cos must be greater than or equal to "
              "this of q. The head_dim of sin and cos must be the same as this "
              "of q. But received sin's "
              "shape is {%s}, q's shape is {%s}.",
              sin_dims,
              q.dims()));

      auto position_ids_dims = position_ids.get_ptr()->dims();
      PADDLE_ENFORCE_EQ(position_ids_dims.size(),
                        2,
                        phi::errors::InvalidArgument(
                            "The dims of position_ids is expected to "
                            "be 2, but received %d.",
                            position_ids_dims.size()));

      PADDLE_ENFORCE_EQ(
          (position_ids_dims[0] == batch_size &&
           position_ids_dims[1] == seq_len),
          true,
          phi::errors::InvalidArgument(
              "The batch_size and seq_len of position_ids must be the same as "
              "those of q. But received position_ids's "
              "shape is {%s}, q's shape is {%s}.",
              position_ids_dims,
              q.dims()));

      position_ids_data = position_ids->data<int64_t>();
    } else {
      PADDLE_ENFORCE_EQ(
          (sin_dims[dims_size - 1] == head_dim &&
           sin_dims[sin_seq_len_dim] == seq_len),
          true,
          phi::errors::InvalidArgument(
              "The seq_len and head_dim of sin and cos "
              "must be the same as those of q. But received sin's "
              "shape is {%s}, q's shape is {%s}.",
              sin_dims,
              q.dims()));
    }

    sin_cos_data[0] = sin->data<T>();
    sin_cos_data[1] = cos->data<T>();

    flag_sin_cos = true;
  }

  int64_t num_heads_kv = GetNumHeadsOfKV(k, v, inputs_num_heads, num_inputs);

  const uint32_t kThreadsPerBlock = 256;
  const uint32_t kWarpSize = 32;
  const uint32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  dim3 grid((uint32_t)batch_size, (uint32_t)seq_len);
  dim3 block(kWarpSize, kWarpsPerBlock);

  int sign = 1;

  VectorizedFusedRopeKernel<T, MPType, vec_size>
      <<<grid, block, 0, dev_ctx.stream()>>>(ins_data,
                                             sin_cos_data,
                                             position_ids_data,
                                             batch_size,
                                             seq_len,
                                             inputs_num_heads[0],
                                             num_heads_kv,
                                             head_dim,
                                             batch_stride_q,
                                             seq_stride_q,
                                             batch_stride_kv,
                                             seq_stride_kv,
                                             outs_data,
                                             div_c,
                                             use_neox_rotary_style,
                                             flag_sin_cos,
                                             sign,
                                             num_inputs);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
