// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void SequenceUnpadXPUKernel(const Context& ctx,
                            const DenseTensor& x,
                            const DenseTensor& length,
                            DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_dims = x.dims();
  auto len_dims = length.dims();
  auto* seq_len_ptr = length.data<int64_t>();
  int64_t batch_size = len_dims[0];
  std::vector<uint64_t> out_lod0(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    out_lod0[i + 1] = out_lod0[i] + seq_len_ptr[i];
  }
  phi::LegacyLoD out_lod;
  out_lod.push_back(out_lod0);

  int64_t out_dim0 = out_lod0.back();
  std::vector<int64_t> out_dims{out_dim0};
  if (x_dims.size() == 2) {
    out_dims.push_back(1);
  } else {
    for (int i = 2; i < x_dims.size(); ++i) {
      out_dims.push_back(x_dims[i]);
    }
  }
  out->Resize(phi::make_ddim(out_dims));
  out->set_lod(out_lod);
  XPUType* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  std::vector<int> lod_cpu(out_lod0.begin(), out_lod0.end());
  xpu::VectorParam<int> query_lod = {
      lod_cpu.data(), (int64_t)lod_cpu.size(), nullptr};
  int64_t dim = out->numel() / out_dim0;
  int r = xpu::sequence_unpad<XPUType, int>(
      ctx.x_context(),                               /* ctx */
      reinterpret_cast<const XPUType*>(x.data<T>()), /* pad_data */
      out_data,                                      /* seq_data */
      query_lod,                                     /* sequence */
      x_dims[1],                                     /* pad_seq_len */
      dim /* dim */);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sequence_unpad_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(sequence_unpad_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::SequenceUnpadXPUKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
}
