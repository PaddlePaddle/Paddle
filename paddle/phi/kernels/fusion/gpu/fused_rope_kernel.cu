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
                     const optional<DenseTensor>& k,
                     const optional<DenseTensor>& v,
                     const optional<DenseTensor>& sin,
                     const optional<DenseTensor>& cos,
                     const optional<DenseTensor>& position_ids,
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
  auto freqs_head_dim = head_dim;

  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  auto stream = dev_ctx.stream();
  const T* sin_data = sin.get_ptr() ? sin.get_ptr()->data<T>() : nullptr;
  const T* cos_data = cos.get_ptr() ? cos.get_ptr()->data<T>() : nullptr;
  const int64_t* position_ids_data =
      position_ids.get_ptr() ? position_ids.get_ptr()->data<int64_t>()
                             : nullptr;

  bool flag_sin_cos = false;

  if (sin_data && cos_data) {
    PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                      cos.get_ptr()->dims(),
                      common::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin.get_ptr()->dims(),
                          cos.get_ptr()->dims()));

    auto sin_dims = sin.get_ptr()->dims();
    int dims_size = sin_dims.size();
    PADDLE_ENFORCE_EQ((dims_size == 2 || dims_size == 4),
                      true,
                      common::errors::InvalidArgument(
                          "The dims of sin and cos is expected to "
                          "be 2 or 4, but received %d.",
                          dims_size));
    if (dims_size == 4) {
      // sin.shape: [1, seq_len, 1, head_dim]
      PADDLE_ENFORCE_EQ(
          (sin_dims[0] == 1 && sin_dims[2] == 1),
          true,
          common::errors::InvalidArgument(
              "The batch_size and num_heads of sin and cos must be 1."));
    }
    int sin_seq_len_dim = (dims_size) == 4 ? 1 : 0;

    if (position_ids_data) {
      auto position_ids_dims = position_ids.get_ptr()->dims();
      PADDLE_ENFORCE_EQ(position_ids_dims.size(),
                        2,
                        common::errors::InvalidArgument(
                            "The dims of position_ids is expected to "
                            "be 2, but received %d.",
                            position_ids_dims.size()));

      PADDLE_ENFORCE_EQ(
          (position_ids_dims[0] == batch_size &&
           position_ids_dims[1] == seq_len),
          true,
          common::errors::InvalidArgument(
              "The batch_size and seq_len of position_ids must be the same as "
              "those of q. But received position_ids's "
              "shape is {%s}, q's shape is {%s}.",
              position_ids_dims,
              q.dims()));
    }

    freqs_head_dim = sin_dims[dims_size - 1];
    flag_sin_cos = true;
  }

  // Q
  int64_t stride_s_q = time_major ? q.strides()[0] : q.strides()[1];
  int64_t stride_b_q = time_major ? q.strides()[1] : q.strides()[0];
  int64_t stride_h_q = q.strides()[2];
  int64_t stride_d_q = q.strides()[3];

  int64_t o_stride_s_q = time_major ? out_q->strides()[0] : out_q->strides()[1];
  int64_t o_stride_b_q = time_major ? out_q->strides()[1] : out_q->strides()[0];
  int64_t o_stride_h_q = out_q->strides()[2];
  int64_t o_stride_d_q = out_q->strides()[3];

  FusedRopeKernelLauncher(q.data<T>(),
                          sin_data,
                          cos_data,
                          out_q->data<T>(),
                          FusedRopeKernelImpl<T, int>,
                          FusedRopeKernelImpl<T, int64_t>,
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
  int k_num_heads = -1;
  if (k) {
    k_num_heads = k->dims()[2];
    auto k_batch_size = time_major ? k->dims()[1] : k->dims()[0];
    PADDLE_ENFORCE_LE(
        batch_size,
        k_batch_size,
        common::errors::InvalidArgument("The batch_size of q (%d) must be less "
                                        "than or equal to k's (%d).",
                                        batch_size,
                                        k_batch_size));

    int64_t stride_s_k = time_major ? k->strides()[0] : k->strides()[1];
    int64_t stride_b_k = time_major ? k->strides()[1] : k->strides()[0];
    int64_t stride_h_k = k->strides()[2];
    int64_t stride_d_k = k->strides()[3];

    int64_t o_stride_s_k =
        time_major ? out_k->strides()[0] : out_k->strides()[1];
    int64_t o_stride_b_k =
        time_major ? out_k->strides()[1] : out_k->strides()[0];
    int64_t o_stride_h_k = out_k->strides()[2];
    int64_t o_stride_d_k = out_k->strides()[3];

    FusedRopeKernelLauncher(k->data<T>(),
                            sin_data,
                            cos_data,
                            out_k->data<T>(),
                            FusedRopeKernelImpl<T, int>,
                            FusedRopeKernelImpl<T, int64_t>,
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
                            k->numel(),
                            stream);
  }

  // V
  if (v) {
    auto v_num_heads = v->dims()[2];
    // Multi Query Attention (MQA) or Group Query Attention (GQA)
    if (k_num_heads != -1) {
      PADDLE_ENFORCE_EQ(
          k_num_heads == v_num_heads,
          true,
          common::errors::InvalidArgument(
              "The num_heads of k must be equal to the num_heads of v when v "
              "is not none."
              "But received num_heads of k is %d, num_heads of v is %d",
              k_num_heads,
              v_num_heads));
    }
    PADDLE_ENFORCE_EQ(
        num_heads == v_num_heads ||
            num_heads != v_num_heads && num_heads % v_num_heads == 0,
        true,
        common::errors::InvalidArgument(
            "The MQA or GQA mode is entered, when the number of heads of qkv "
            "is not exactly the same two by two. This mode requires "
            "num_heads of q to be divisible by k,v."
            "But received num_heads of q is %d, num_heads of k,v is %d",
            num_heads,
            v_num_heads));
    auto v_batch_size = time_major ? v->dims()[1] : v->dims()[0];
    PADDLE_ENFORCE_LE(
        batch_size,
        v_batch_size,
        common::errors::InvalidArgument("The batch_size of q (%d) must be less "
                                        "than or equal to v's (%d).",
                                        batch_size,
                                        v_batch_size));

    int64_t stride_s_v = time_major ? v->strides()[0] : v->strides()[1];
    int64_t stride_b_v = time_major ? v->strides()[1] : v->strides()[0];
    int64_t stride_h_v = v->strides()[2];
    int64_t stride_d_v = v->strides()[3];

    int64_t o_stride_s_v =
        time_major ? out_v->strides()[0] : out_v->strides()[1];
    int64_t o_stride_b_v =
        time_major ? out_v->strides()[1] : out_v->strides()[0];
    int64_t o_stride_h_v = out_v->strides()[2];
    int64_t o_stride_d_v = out_v->strides()[3];

    FusedRopeKernelLauncher(v->data<T>(),
                            sin_data,
                            cos_data,
                            out_v->data<T>(),
                            FusedRopeKernelImpl<T, int>,
                            FusedRopeKernelImpl<T, int64_t>,
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
                            v->numel(),
                            stream);
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
