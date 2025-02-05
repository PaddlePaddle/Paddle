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
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void TensorUnfoldKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        int64_t axis,
                        int64_t size,
                        int64_t step,
                        DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  if (axis < 0) {
    axis += input.dims().size();
  }

  const DDim& input_dims = input.dims();
  const DDim& input_stride = input.strides();
  int64_t max_size =
      input_dims.size() == 0 ? 1 : input_dims[static_cast<int>(axis)];

  PADDLE_ENFORCE_LE(
      size,
      max_size,
      common::errors::InvalidArgument(
          "paddle.unfold size(%d) must be less than shape[axis](%d).",
          size,
          max_size));
  PADDLE_ENFORCE_GT(step,
                    0,
                    common::errors::InvalidArgument(
                        "paddle.unfold step must be greater than 0"));

  std::vector<int64_t> shape(input_dims.size() + 1);
  std::vector<int64_t> stride(input_dims.size() + 1);

  shape[input_dims.size()] = size;
  stride[input_dims.size()] =
      input_dims.size() == 0 ? 1 : input_stride[static_cast<int>(axis)];
  for (int i = 0; i < input_dims.size(); ++i) {
    if (i == axis) {
      shape[i] = (input_dims[i] - size) / step + 1;
      stride[i] = step * input_stride[i];
    } else {
      shape[i] = input_dims[i];
      stride[i] = input_stride[i];
    }
  }

  auto meta = out->meta();
  meta.dims = DDim(shape.data(), static_cast<int>(shape.size()));
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  meta.offset = input.offset();
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(tensor_unfold,
                                         STRIDED,
                                         phi::TensorUnfoldKernel) {}
