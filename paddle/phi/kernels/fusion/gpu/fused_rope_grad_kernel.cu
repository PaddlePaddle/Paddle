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
                         float rotary_emb_base,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  int64_t numel = dout_q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(dq);

  phi::Array<int64_t, 3> inputs_num_heads;
  // small size for broadcast
  auto batch_size = time_major ? dout_q.dims()[1] : dout_q.dims()[0];
  auto seq_len = time_major ? dout_q.dims()[0] : dout_q.dims()[1];
  inputs_num_heads[0] = dout_q.dims()[2];
  auto head_dim = dout_q.dims()[3];
  PADDLE_ENFORCE_NE(head_dim % 2,
                    1,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  constexpr const int vec_size = 2;

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);

  int64_t grid = config.block_per_grid.x;
  int64_t block = config.thread_per_block.x;
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
    num_inputs++;
  }

  if (dout_v) {
    dev_ctx.template Alloc<T>(dv);
    outs_data[num_inputs] = dv->data<T>();
    ins_data[num_inputs] = dout_v->data<T>();
    inputs_num_heads[num_inputs] = dv->dims()[2];
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

  bool is_same_num_heads = true;
  auto prev_num_heads = inputs_num_heads[0];
  for (int i = 1; i < num_inputs; ++i) {
    if (prev_num_heads != inputs_num_heads[i]) {
      is_same_num_heads = false;
      break;
    }
    prev_num_heads = inputs_num_heads[i];
  }

  int sign = -1;

  VectorizedFusedRopeCudaKernelFunc<T, MPType, vec_size> kernel_func =
      use_neox_rotary_style
          ? VectorizedFusedRopeWithRotateEveryTwoKernel<T, MPType, vec_size>
          : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, vec_size>;

  if (is_same_num_heads) {
    int64_t batch_stride =
        time_major ? dout_q.strides()[1] : dout_q.strides()[0];
    int64_t seq_stride = time_major ? dout_q.strides()[0] : dout_q.strides()[1];
    kernel_func<<<grid, block, 0, stream>>>(ins_data,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[0],
                                            head_dim,
                                            batch_stride,
                                            seq_stride,
                                            num_inputs,
                                            div_c,
                                            rotary_emb_base,
                                            outs_data);

  } else {
    // rotary position embedding Q
    int64_t batch_stride_q =
        time_major ? dout_q.strides()[1] : dout_q.strides()[0];
    int64_t seq_stride_q =
        time_major ? dout_q.strides()[0] : dout_q.strides()[1];
    kernel_func<<<grid, block, 0, stream>>>(ins_data,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[0],
                                            head_dim,
                                            batch_stride_q,
                                            seq_stride_q,
                                            1,
                                            div_c,
                                            rotary_emb_base,
                                            outs_data);

    // rotary position embedding K,V
    int64_t batch_stride_kv = time_major
                                  ? inputs_num_heads[1] * head_dim
                                  : seq_len * inputs_num_heads[1] * head_dim;
    int64_t seq_stride_kv = time_major
                                ? batch_size * inputs_num_heads[1] * head_dim
                                : inputs_num_heads[1] * head_dim;

    phi::Array<const T*, 3> input_kv{ins_data[1], ins_data[2], nullptr};
    phi::Array<T*, 3> out_kv{outs_data[1], outs_data[2], nullptr};
    kernel_func<<<grid, block, 0, stream>>>(input_kv,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[1],
                                            head_dim,
                                            batch_stride_kv,
                                            seq_stride_kv,
                                            num_inputs - 1,
                                            div_c,
                                            rotary_emb_base,
                                            out_kv);
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
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
