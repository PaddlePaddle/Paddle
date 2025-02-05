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
#include "paddle/phi/kernels/view_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename Context>
void ViewShapeKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const std::vector<int64_t>& dims,
                     DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(input, dims, &meta_out);
  auto meta = input.meta();
  meta.offset = input.offset();
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(view_shape,
                                         ALL_LAYOUT,
                                         phi::ViewShapeKernel) {}
