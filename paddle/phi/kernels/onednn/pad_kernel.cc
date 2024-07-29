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

#include "paddle/phi/kernels/pad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/pad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void PadKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int>& paddings,
               const Scalar& pad_value,
               bool pad_from_first_axis,
               DenseTensor* out) {
  // pad the length of paddings to 2*x.ndim
  auto x_dim = x.dims();
  std::vector<int> pad(2 * x_dim.size());
  int paddings_len = paddings.size();
  for (size_t i = 0; i < pad.size(); ++i) {
    int pad_i = static_cast<int>(i) < paddings_len ? paddings[i] : 0;
    pad[i] = pad_i;
  }

  if ((static_cast<int>(paddings_len) == x_dim.size() * 2) &&
      pad_from_first_axis) {
    std::vector<int64_t> copied_paddings(pad.begin(), pad.end());

    std::swap(copied_paddings[0], copied_paddings[2]);
    std::swap(copied_paddings[1], copied_paddings[3]);
    PadOpKernel<T, Context>(
        dev_ctx, x, copied_paddings, pad_value.to<float>(), out);
  } else {
    std::vector<int> pad_reversed(2 * x_dim.size());
    for (int i = 2 * x_dim.size() - 1; i >= 0; --i) {
      int index = 2 * x_dim.size() - 1 - i;
      pad_reversed[i] = (index % 2 == 1) ? pad[index - 1] : pad[index + 1];
    }
    std::vector<int64_t> copied_paddings(pad_reversed.begin(),
                                         pad_reversed.end());

    std::swap(copied_paddings[0], copied_paddings[2]);
    std::swap(copied_paddings[1], copied_paddings[3]);
    PadOpKernel<T, Context>(
        dev_ctx, x, copied_paddings, pad_value.to<float>(), out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(pad, OneDNN, ONEDNN, phi::PadKernel, float) {}
