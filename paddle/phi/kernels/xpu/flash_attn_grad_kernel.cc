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

#include "paddle/phi/kernels/flash_attn_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#ifdef PADDLE_WITH_XPU_XHPC
#include "xfa/flash_api.h"
#endif

namespace phi {

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
#ifdef PADDLE_WITH_XPU_XHPC
  if (causal == false) {
    PADDLE_THROW(phi::errors::Unimplemented("causal should be true"));
  }

  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  // raw pointers
  using XPUType = typename XPUTypeTrait<T>::Type;
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  const XPUType* out_data = reinterpret_cast<const XPUType*>(out.data<T>());
  const float* softmax_lse_data = softmax_lse.data<float>();
  const XPUType* dout_data = reinterpret_cast<const XPUType*>(dout.data<T>());

  // output
  XPUType* dq_data = reinterpret_cast<XPUType*>(dq->data<T>());
  XPUType* dk_data = reinterpret_cast<XPUType*>(dk->data<T>());
  XPUType* dv_data = reinterpret_cast<XPUType*>(dv->data<T>());

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

  // template<typename T, typename TACCUM, typename TGEMM, typename TID = int>
  // int mha_varlen_bwd(xdnn::Context* ctx, const T* dout, const T* q, const T*
  // k, const T* v, const T* out, const TACCUM* softmax_lse, T* dq, T* dk, T*
  // dv, const xdnn::VectorParam<TID>& lod_seqlens_q, const
  // xdnn::VectorParam<TID>& lod_seqlens_k, int64_t max_seqlen_q, int64_t
  // max_seqlen_k, int64_t head_num, int64_t head_num_k, int64_t head_dim, const
  // float softmax_scale = 0.0f, const float p_dropout = 0.0f, int seed =
  // 0x45678901, const bool is_causal = true, const TACCUM* attn_mask = nullptr,
  // const TACCUM* bias = nullptr, const float* q_maxptr = nullptr, const float*
  // k_maxptr = nullptr, const float* v_maxptr = nullptr, const float* o_maxptr
  // = nullptr, float* dq_maxptr = nullptr, float* dk_maxptr = nullptr, float*
  // dv_maxptr = nullptr, const float* do_maxptr = nullptr);
  int r = baidu::xpu::xfa::mha_varlen_bwd<XPUType, float, tfloat32, int>(
      ctx.x_context(),
      dout_data,                    // dout
      q_data,                       // q
      k_data,                       // k
      v_data,                       // v
      out_data,                     // out
      softmax_lse_data,             // softmax_lse
      dq_data,                      // dq
      dk_data,                      // dk
      dv_data,                      // dv
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
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_bwd");
#else
  PADDLE_THROW(phi::errors::PreconditionNotMet(
      "re-compile using -DWITH_XPU_XHPC=ON to use FlashAttnGradKernel"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::bfloat16,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
