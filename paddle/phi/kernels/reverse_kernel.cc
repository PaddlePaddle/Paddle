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

#include "paddle/phi/kernels/reverse_kernel.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void ReverseArrayKernel(const Context& dev_ctx,
                        const TensorArray& x,
                        const IntArray& axis,
                        TensorArray* out) {
  PADDLE_ENFORCE_EQ(
      x.size(),
      out->size(),
      phi::errors::InvalidArgument("The input size(%d) and output size(%d) of "
                                   "ReverseArrayKernel is different.",
                                   x.size(),
                                   out->size()));
  for (size_t offset = 0; offset < x.size(); ++offset) {
    auto& x_tensor = x.at(offset);
    PADDLE_ENFORCE_GT(
        x_tensor.memory_size(),
        0,
        phi::errors::PreconditionNotMet(
            "The input LoDTensorArray X[%d] holds no memory.", offset));
    auto out_offset = x.size() - offset - 1;
    auto& out_tensor = out->at(out_offset);

    out_tensor.set_lod(x_tensor.lod());
    phi::Copy<Context>(
        dev_ctx, x_tensor, dev_ctx.GetPlace(), false, &out_tensor);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(reverse_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReverseArrayKernel,
                   int,
                   uint8_t,
                   int64_t,
                   bool,
                   float,
                   double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(reverse_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReverseArrayKernel,
                   int,
                   uint8_t,
                   int64_t,
                   bool,
                   float,
                   double) {}

#endif
