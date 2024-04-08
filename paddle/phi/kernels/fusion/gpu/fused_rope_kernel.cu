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
  auto batch_size = time_major ? q.dims()[1] : q.dims()[0];
  auto seq_len = time_major ? q.dims()[0] : q.dims()[1];
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
  auto sin_dims = sin.get_ptr()->dims();
  if (sin.get_ptr() && cos.get_ptr()) {
    PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                      cos.get_ptr()->dims(),
                      phi::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin.get_ptr()->dims(),
                          cos.get_ptr()->dims()));

    int dims_size = sin_dims.size();
    PADDLE_ENFORCE_EQ(
        (dims_size == 2 || dims_size == 4),
        true,
        phi::errors::InvalidArgument("The dims of sin and cos is expected to "
                                     "be 2 or 4, but received %d.",
                                     dims_size));
    if (dims_size == 4) {
      // sin.shape: [batch_size, seq_len, 1, head_dim]
      PADDLE_ENFORCE_EQ(
          (sin_dims[0] == 1 || sin_dims[0] == batch_size),
          true,
          phi::errors::InvalidArgument("The batch_size of sin and cos must be "
                                       "1 or equal to batch_size."));
      PADDLE_ENFORCE_EQ((sin_dims[2] == 1),
                        true,
                        phi::errors::InvalidArgument(
                            "The num_heads of sin and cos must be 1."));
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
  VectorizedFusedRopeCudaKernelFunc<T, MPType, vec_size> kernel_func =
      use_neox_rotary_style
          ? VectorizedFusedRopeWithRotateEveryTwoKernel<T, MPType, vec_size>
          : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, vec_size>;

  if (is_same_num_heads) {
    int64_t batch_stride = time_major ? q.strides()[1] : q.strides()[0];
    int64_t seq_stride = time_major ? q.strides()[0] : q.strides()[1];
    kernel_func<<<grid, block, 0, stream>>>(ins_data,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            sin_dims[0],
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[0],
                                            head_dim,
                                            batch_stride,
                                            seq_stride,
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
    // rotary position embedding Q
    int64_t batch_stride_q = time_major ? q.strides()[1] : q.strides()[0];
    int64_t seq_stride_q = time_major ? q.strides()[0] : q.strides()[1];

    kernel_func<<<grid, block, 0, stream>>>(ins_data,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            sin_dims[0],
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[0],
                                            head_dim,
                                            batch_stride_q,
                                            seq_stride_q,
                                            outs_data,
                                            1,
                                            div_c);

    // rotary position embedding K,V
    phi::Array<const T*, 3> input_kv{ins_data[1], ins_data[2], nullptr};
    phi::Array<T*, 3> out_kv{outs_data[1], outs_data[2], nullptr};
    int64_t batch_stride_kv = time_major
                                  ? inputs_num_heads[1] * head_dim
                                  : seq_len * inputs_num_heads[1] * head_dim;
    int64_t seq_stride_kv = time_major
                                ? batch_size * inputs_num_heads[1] * head_dim
                                : inputs_num_heads[1] * head_dim;

    kernel_func<<<grid, block, 0, stream>>>(input_kv,
                                            sin_cos_data,
                                            position_ids_data,
                                            flag_sin_cos,
                                            sign,
                                            sin_dims[0],
                                            batch_size,
                                            seq_len,
                                            inputs_num_heads[1],
                                            head_dim,
                                            batch_stride_kv,
                                            seq_stride_kv,
                                            out_kv,
                                            num_inputs - 1,
                                            div_c);
  }
}

template <typename T, typename Context>
void FusedRope3DKernel(const Context& dev_ctx,
                       const DenseTensor& q,
                       const paddle::optional<DenseTensor>& k,
                       const paddle::optional<DenseTensor>& v,
                       const paddle::optional<DenseTensor>& sin,
                       const paddle::optional<DenseTensor>& cos,
                       DenseTensor* out_q,
                       DenseTensor* out_k,
                       DenseTensor* out_v) {
  dev_ctx.template Alloc<T>(out_q);

  phi::Array<int64_t, 3> inputs_num_heads;

  // q.shape: [batch_size, seq_len, num_heads, head_dim]
  auto batch_size = q.dims()[0];
  auto seq_len = q.dims()[1];
  inputs_num_heads[0] = q.dims()[2];
  auto head_dim = q.dims()[3];

  PADDLE_ENFORCE_EQ(head_dim % 6,
                    0,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 6."));

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  constexpr const int vec_size = 2;
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, q.numel(), vec_size);

  int64_t grid = config.block_per_grid.x;
  int64_t block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();

  phi::Array<T*, 3> outs_data;
  phi::Array<const T*, 3> ins_data;
  phi::Array<const T*, 2> sin_cos_data;

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

  PADDLE_ENFORCE_EQ(
      sin.get_ptr() && cos.get_ptr(),
      true,
      phi::errors::InvalidArgument("The sin and cos should not be None."));

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
      (dims_size == 4 || dims_size == 6),
      true,
      phi::errors::InvalidArgument("The dims of sin and cos is expected to "
                                   "be 4 or 6, but received %d.",
                                   dims_size));
  // sin.shape: [1 or batch_size, frame * height * width, 1, head_dim] or
  // sin.shape: [1 or batch_size, frame, height, width, 1, head_dim]
  PADDLE_ENFORCE_EQ(
      (sin_dims[0] == 1 || sin_dims[0] == batch_size),
      true,
      phi::errors::InvalidArgument(
          "The batch_size and num_heads of sin and cos must be 1."));
  PADDLE_ENFORCE_EQ(
      (sin_dims[dims_size - 2] == 1),
      true,
      phi::errors::InvalidArgument(
          "The batch_size and num_heads of sin and cos must be 1."));
  if (dims_size == 4) {
    PADDLE_ENFORCE_EQ((sin_dims[1] == seq_len),
                      true,
                      phi::errors::InvalidArgument(
                          "The sin.shape[1] must be equal to seq_len."));
  }
  if (dims_size == 6) {
    PADDLE_ENFORCE_EQ((sin_dims[1] * sin_dims[2] * sin_dims[3] == seq_len),
                      true,
                      phi::errors::InvalidArgument(
                          "The sin.shape[1] * sin.shape[2] * sin.shape[3] "
                          "must be equal to seq_len."));
  }
  PADDLE_ENFORCE_EQ((sin_dims[dims_size - 1] == head_dim),
                    true,
                    phi::errors::InvalidArgument(
                        "The head_dim of sin and cos "
                        "must be the same as that of q. But received sin's "
                        "shape is {%s}, q's shape is {%s}.",
                        sin_dims,
                        q.dims()));
  sin_cos_data[0] = sin->data<T>();
  sin_cos_data[1] = cos->data<T>();

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
    int64_t seq_stride = q.strides()[1];
    VectorizedFusedRope3DKernel<T, MPType, vec_size>
        <<<grid, block, 0, stream>>>(ins_data,
                                     sin_cos_data,
                                     sin_dims[0],
                                     sign,
                                     batch_size,
                                     seq_len,
                                     inputs_num_heads[0],
                                     head_dim,
                                     seq_stride,
                                     outs_data,
                                     num_inputs);
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
    // rotary position embedding Q
    int64_t seq_stride_q = q.strides()[1];
    VectorizedFusedRope3DKernel<T, MPType, vec_size>
        <<<grid, block, 0, stream>>>(ins_data,
                                     sin_cos_data,
                                     sin_dims[0],
                                     sign,
                                     batch_size,
                                     seq_len,
                                     inputs_num_heads[0],
                                     head_dim,
                                     seq_stride_q,
                                     outs_data,
                                     1);

    // rotary position embedding K,V
    phi::Array<const T*, 3> input_kv{ins_data[1], ins_data[2], nullptr};
    phi::Array<T*, 3> out_kv{outs_data[1], outs_data[2], nullptr};
    int64_t seq_stride_kv = inputs_num_heads[1] * head_dim;

    VectorizedFusedRope3DKernel<T, MPType, vec_size>
        <<<grid, block, 0, stream>>>(input_kv,
                                     sin_cos_data,
                                     sin_dims[0],
                                     sign,
                                     batch_size,
                                     seq_len,
                                     inputs_num_heads[1],
                                     head_dim,
                                     seq_stride_kv,
                                     out_kv,
                                     num_inputs - 1);
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
PD_REGISTER_KERNEL(fused_rotary_position_embedding_3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRope3DKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
