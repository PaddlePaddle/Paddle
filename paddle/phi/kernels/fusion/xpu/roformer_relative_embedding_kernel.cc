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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void RoformerRelativePosXPUKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& sin_emb,
                                  const DenseTensor& cos_emb,
                                  int max_pos_len,
                                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* sin_emb_data = sin_emb.data<float>();
  auto* cos_emb_data = cos_emb.data<float>();
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  auto x_dims = x.dims();
  int batch = x_dims[0];
  int head_num = x_dims[1];
  int seqlen = x_dims[2];
  int head_dim = x_dims[3];
  if (seqlen > max_pos_len) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The input sequence length should be less than or equal to the "
        "maximum position length. But received seqlen: %d, max_pos_len: %d",
        seqlen,
        max_pos_len));
  }
  std::vector<int> lod;
  lod.resize(batch + 1);
  for (int i = 0; i < batch + 1; i++) {
    lod[i] = i * seqlen;
  }
  int r =
      xpu::rope<XPUType>(ctx.x_context(),
                         x_data,
                         out_data,
                         cos_emb_data,
                         sin_emb_data,
                         batch,
                         head_num,
                         head_dim,
                         head_num * head_dim,
                         lod,
                         max_pos_len,
                         false,  // no vsl
                         true);  // transpose to [n, seql, head_num, head_dim]

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "roformer_relative_embedding_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(roformer_relative_embedding_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::RoformerRelativePosXPUKernel,
                   float,
                   phi::dtype::float16) {}
