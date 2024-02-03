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
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  int64_t numel = q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(out_q);

  phi::Array<int64_t, 3> inputs_num_heads;

  // q.shape: [batch_size, seq_len, num_heads, head_dim]
  auto batch_size = q.dims()[0];
  auto seq_len = q.dims()[1];
  inputs_num_heads[0] = q.dims()[2];
  auto head_dim = q.dims()[3];

  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    phi::errors::InvalidArgument(
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

  ins_data[0] = q.data<T>();
  outs_data[0] = out_q->data<T>();
  int num_inputs = 1;

  if (k) {
    dev_ctx.template Alloc<T>(out_k);
    ins_data[num_inputs] = k->data<T>();
    outs_data[num_inputs] = out_k->data<T>();
    inputs_num_heads[num_inputs] = k->dims()[2];
    num_inputs++;
  }

  if (v) {
    dev_ctx.template Alloc<T>(out_v);
    ins_data[num_inputs] = v->data<T>();
    outs_data[num_inputs] = out_v->data<T>();
    inputs_num_heads[num_inputs] = v->dims()[2];
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

  bool is_same_num_heads = true;
  auto prev_num_heads = inputs_num_heads[0];
  for (int i = 1; i < num_inputs; ++i) {
    if (prev_num_heads != inputs_num_heads[i]) {
      is_same_num_heads = false;
      break;
    }
    prev_num_heads = inputs_num_heads[i];
  }

  int sign = 1;
  if (is_same_num_heads) {
    VectorizedFusedRopeCudaKernelFunc<T, MPType, 3, vec_size> kernel_func_qkv =
        use_neox_rotary_style
            ? VectorizedFusedRopeWithRotateEveryTwoKernel<T,
                                                          MPType,
                                                          3,
                                                          vec_size>
            : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, 3, vec_size>;

    kernel_func_qkv<<<grid, block, 0, stream>>>(ins_data,
                                                sin_cos_data,
                                                position_ids_data,
                                                flag_sin_cos,
                                                sign,
                                                batch_size,
                                                seq_len,
                                                inputs_num_heads[0],
                                                head_dim,
                                                outs_data,
                                                num_inputs,
                                                div_c);
  } else {
    // Multi Query Attention (MQA) or Group Query Attention (GQA)
    PADDLE_ENFORCE_EQ(
        (inputs_num_heads[0] != inputs_num_heads[num_inputs - 1]) &&
            (inputs_num_heads[0] % inputs_num_heads[num_inputs - 1] == 0),
        true,
        phi::errors::InvalidArgument(
            "The MQA or GQA mode is entered, when the number of heads of qkv "
            "is not exactly the same two by two. This mode requires "
            "num_heads of q to be divisible by k,v."
            "But recieved num_heads of q is %d, num_heads of k,v is %d",
            inputs_num_heads[0],
            inputs_num_heads[num_inputs - 1]));

    if (k.get_ptr() && v.get_ptr()) {
      PADDLE_ENFORCE_EQ(
          inputs_num_heads[1] == inputs_num_heads[2],
          true,
          phi::errors::InvalidArgument(
              "The num_heads of k must be equal to the num_heads of v when v "
              "is not none."
              "But recieved num_heads of k is %d, num_heads of v is %d",
              inputs_num_heads[1],
              inputs_num_heads[2]));
    }

    VectorizedFusedRopeCudaKernelFunc<T, MPType, 1, vec_size> kernel_func_q =
        use_neox_rotary_style
            ? VectorizedFusedRopeWithRotateEveryTwoKernel<T,
                                                          MPType,
                                                          1,
                                                          vec_size>
            : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, 1, vec_size>;
    VectorizedFusedRopeCudaKernelFunc<T, MPType, 2, vec_size> kernel_func_kv =
        use_neox_rotary_style
            ? VectorizedFusedRopeWithRotateEveryTwoKernel<T,
                                                          MPType,
                                                          2,
                                                          vec_size>
            : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, 2, vec_size>;

    // rotary position embedding Q
    phi::Array<const T*, 1> input_q{ins_data[0]};
    phi::Array<T*, 1> out_q{outs_data[0]};
    kernel_func_q<<<grid, block, 0, stream>>>(input_q,
                                              sin_cos_data,
                                              position_ids_data,
                                              flag_sin_cos,
                                              sign,
                                              batch_size,
                                              seq_len,
                                              inputs_num_heads[0],
                                              head_dim,
                                              out_q,
                                              1,
                                              div_c);

    // rotary position embedding K,V
    phi::Array<const T*, 2> input_kv{ins_data[1], ins_data[2]};
    phi::Array<T*, 2> out_kv{outs_data[1], outs_data[2]};
    kernel_func_kv<<<grid, block, 0, stream>>>(input_kv,
                                               sin_cos_data,
                                               position_ids_data,
                                               flag_sin_cos,
                                               sign,
                                               batch_size,
                                               seq_len,
                                               inputs_num_heads[1],
                                               head_dim,
                                               out_kv,
                                               num_inputs - 1,
                                               div_c);
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
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
