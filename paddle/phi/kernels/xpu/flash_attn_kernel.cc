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

#include "paddle/phi/kernels/flash_attn_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#ifdef PADDLE_WITH_XPU_XHPC
#include "xfa/flash_api.h"
#endif

namespace phi {

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const paddle::optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_XPU_XHPC
  if (causal == false) {
    PADDLE_THROW(phi::errors::Unimplemented("causal should be true"));
  }
  if (return_softmax == true) {
    PADDLE_THROW(phi::errors::Unimplemented("return_softmax should be false"));
  }

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  // lod info
  std::vector<int> qlod_vec = {0};
  std::vector<int> kvlod_vec = {0};
  for (int batch_idx = 1; batch_idx <= batch_size; ++batch_idx) {
    qlod_vec.push_back(seqlen_q * batch_idx);
    kvlod_vec.push_back(seqlen_k * batch_idx);
  }
  api::VectorParam<int> qlod{
      qlod_vec.data(), static_cast<int64_t>(qlod_vec.size()), nullptr};
  api::VectorParam<int> kvlod{
      kvlod_vec.data(), static_cast<int64_t>(kvlod_vec.size()), nullptr};

  // output: softmax_lse
  std::vector<int64_t> softmax_lse_dims;
  softmax_lse_dims = {batch_size, num_heads, seqlen_q};
  softmax_lse->Resize(phi::make_ddim(softmax_lse_dims));
  ctx.template Alloc<float>(softmax_lse);

  // output: o
  ctx.template Alloc<T>(out);

  // raw pointers
  using XPUType = typename XPUTypeTrait<T>::Type;
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  XPUType* out_data = reinterpret_cast<XPUType*>(out->data<T>());
  float* softmax_lse_data = softmax_lse->data<float>();

  // template <typename T, typename TACCUM, typename TGEMM, typename TID> int
  // mha_varlen_fwd(xdnn::Context* ctx, const T* q, const T* k, const T* v, T*
  // out, TACCUM* softmax_lse, const xdnn::VectorParam<TID>& lod_seqlens_q,
  // const xdnn::VectorParam<TID>& lod_seqlens_k, int64_t max_seqlen_q, int64_t
  // max_seqlen_k, int64_t head_num, int64_t head_num_k, int64_t head_dim, const
  // float softmax_scale = 0.0f, const float p_dropout = 0.0f, int seed =
  // 0x45678901, const bool is_causal = true, const TACCUM* attn_mask = nullptr,
  // const TACCUM* bias = nullptr, const float* q_maxptr = nullptr, const float*
  // k_maxptr = nullptr, const float* v_maxptr = nullptr, float* o_maxptr =
  // nullptr);
  int r = baidu::xpu::xfa::mha_varlen_fwd<XPUType, float, tfloat32, int>(
      ctx.x_context(),
      q_data,                       // q
      k_data,                       // k
      v_data,                       // v
      out_data,                     // out
      softmax_lse_data,             // softmax_lse
      qlod,                         // lod_seqlens_q
      kvlod,                        // lod_seqlens_k
      seqlen_q,                     // max_seqlen_q
      seqlen_k,                     // max_seqlen_k
      num_heads,                    // head_num
      num_heads_k,                  // head_num_k
      head_size,                    // head_dim
      1.0f / std::sqrt(head_size),  // softmax_scale
      dropout                       // p_dropout
  );
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_fwd");
#else
  PADDLE_THROW(phi::errors::PreconditionNotMet(
      "re-compile using -DWITH_XPU_XHPC=ON to use FlashAttnKernel"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::bfloat16,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
