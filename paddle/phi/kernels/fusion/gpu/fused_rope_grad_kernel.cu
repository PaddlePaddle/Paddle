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
                         const paddle::optional<DenseTensor>& sin,
                         const paddle::optional<DenseTensor>& cos,
                         const paddle::optional<DenseTensor>& position_ids,
                         const DenseTensor& dout_q,
                         const paddle::optional<DenseTensor>& dout_k,
                         const paddle::optional<DenseTensor>& dout_v,
                         bool use_neox_rotary_style,
                         bool time_major,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  int64_t numel = dout_q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(dq);

  phi::Array<int64_t, 3> inputs_num_heads;
  // small size for broadcast

  const int kBatchDimIndex = time_major ? 1 : 0;
  const int kSeqlenDimIndex = time_major ? 0 : 1;
  auto batch_size = dout_q.dims()[kBatchDimIndex];
  auto seq_len = dout_q.dims()[kSeqlenDimIndex];
  inputs_num_heads[0] = dout_q.dims()[2];
  auto head_dim = dout_q.dims()[3];

  int64_t batch_stride_q = dout_q.strides()[kBatchDimIndex];
  int64_t seq_stride_q = dout_q.strides()[kSeqlenDimIndex];
  int64_t batch_stride_kv = batch_stride_q;
  int64_t seq_stride_kv = seq_stride_q;

  PADDLE_ENFORCE_NE(head_dim % 2,
                    1,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  constexpr const int vec_size = 2;

  auto stream = dev_ctx.stream();

  phi::Array<T*, 3> outs_data;
  phi::Array<const T*, 3> ins_data;
  phi::Array<const T*, 2> sin_cos_data;
  const int64_t* position_ids_data = NULL;

  ins_data[0] = dout_q.data<T>();
  outs_data[0] = dq->data<T>();
  int num_inputs = 1;

  if (dout_k) {
    dev_ctx.template Alloc<T>(dk);
    outs_data[num_inputs] = dk->data<T>();
    ins_data[num_inputs] = dout_k->data<T>();
    inputs_num_heads[num_inputs] = dk->dims()[2];

    batch_stride_kv = dout_k->strides()[kBatchDimIndex];
    seq_stride_kv = dout_k->strides()[kSeqlenDimIndex];

    num_inputs++;
  }

  if (dout_v) {
    dev_ctx.template Alloc<T>(dv);
    outs_data[num_inputs] = dv->data<T>();
    ins_data[num_inputs] = dout_v->data<T>();
    inputs_num_heads[num_inputs] = dv->dims()[2];

    batch_stride_kv = dout_v->strides()[kBatchDimIndex];
    seq_stride_kv = dout_v->strides()[kSeqlenDimIndex];

    num_inputs++;
  }

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType div_c = static_cast<MPType>(1.0f / head_dim);

  bool flag_sin_cos = false;
  if (sin.get_ptr() && cos.get_ptr()) {
    sin_cos_data[0] = sin->data<T>();
    sin_cos_data[1] = cos->data<T>();

    flag_sin_cos = true;

    if (position_ids) {
      position_ids_data = position_ids->data<int64_t>();
    }
  }

  int64_t num_heads_kv =
      GetNumHeadsOfKV(dout_k, dout_v, inputs_num_heads, num_inputs);

  const uint32_t kThreadsPerBlock = 256;
  const uint32_t kWarpSize = 32;
  const uint32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  dim3 grid((uint32_t)batch_size, (uint32_t)seq_len);
  dim3 block(kWarpSize, kWarpsPerBlock);

  int sign = -1;
  VectorizedFusedRopeKernel<T, MPType, vec_size>
      <<<grid, block, 0, stream>>>(ins_data,
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

PD_REGISTER_KERNEL(fused_rotary_position_embedding_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
