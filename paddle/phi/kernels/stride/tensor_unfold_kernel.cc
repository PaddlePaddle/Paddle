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
#include "paddle/phi/kernels/tensor_unfold_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void TensorUnfoldKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        int64_t axis,
                        int64_t size,
                        int64_t step,
                        DenseTensor* out) {
  if (axis < 0) {
    axis += input.dims().size();
  }

  const DDim& input_dims = input.dims();
  const DDim& input_stride = input.strides();
  int64_t max_size = input_dims.size() == 0 ? 1 : input_dims[axis];

  PADDLE_ENFORCE_LE(
      size,
      max_size,
      phi::errors::InvalidArgument(
          "paddle.unfold size(%d) must be less than shape[axis](%d).",
          size,
          max_size));
  PADDLE_ENFORCE_GT(step,
                    0,
                    phi::errors::InvalidArgument(
                        "paddle.unfold step must be greater than 0"));

  std::vector<int64_t> shape(input_dims.size() + 1);
  std::vector<int64_t> stride(input_dims.size() + 1);

  shape[input_dims.size()] = size;
  stride[input_dims.size()] = input_dims.size() == 0 ? 1 : input_stride[axis];
  for (int i = 0; i < input_dims.size(); ++i) {
    if (i == axis) {
      shape[i] = (input_dims[i] - size) / step + 1;
      stride[i] = step * input_stride[i];
    } else {
      shape[i] = input_dims[i];
      stride[i] = input_stride[i];
    }
  }

  out->Resize(DDim(shape.data(), shape.size()));
  out->set_strides(DDim(stride.data(), stride.size()));
  out->set_offset(input.offset());
  out->ResetHolder(input.Holder());
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    tensor_unfold, STRIDED, phi::TensorUnfoldKernel) {}
