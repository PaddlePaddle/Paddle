// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/shape_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ShapeKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  DDim x_dims = x.dims();

  // Output of shape op is often fed as x to fill_constant ops
  // and we need to rotate a shape otherwise Tensors of wrong shape may be
  // allocated
  if (OneDNNContext::tls().get_cur_paddle_data_layout() == DataLayout::kNHWC &&
      x_dims.size() >= 3) {
    auto rdims = vectorize<int>(x_dims);
    std::rotate(rdims.begin() + 1, rdims.begin() + 2, rdims.end());
    x_dims = make_ddim(rdims);
  }

  out->Resize({x_dims.size()});
  auto out_data = dev_ctx.template Alloc<int32_t>(out);
  for (int i = 0; i < x_dims.size(); ++i) {
    out_data[i] = x_dims[i];
  }

  dnnl::memory::desc out_mem_desc(
      vectorize(out->dims()),
      funcs::ToOneDNNDataType(out->dtype()),
      funcs::GetPlainOneDNNFormat(out->dims().size()));
  out->set_mem_desc(out_mem_desc);
}
}  // namespace phi

PD_REGISTER_KERNEL(shape,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
