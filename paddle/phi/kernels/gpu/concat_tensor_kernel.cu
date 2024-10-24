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

#include "paddle/phi/kernels/concat_tensor_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {
template <typename T>
bool IsConcated(const std::vector<const DenseTensor*>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }
  std::vector<int64_t> numels(tensors.size());
  std::vector<const T*> tensors_data(tensors.size());

  for (size_t i = 0; i < tensors.size(); i++) {
    numels[i] = tensors[i]->numel();
    tensors_data[i] = tensors[i]->data<T>();
  }

  for (size_t i = 0; i < tensors.size() - 1; i++) {
    if (tensors_data[i] + numels[i] != tensors_data[i + 1]) {
      return false;
    }
  }

  return true;
}

template <typename T, typename Context>
void ConcatTensorKernel(const Context& dev_ctx,
                        const std::vector<const DenseTensor*>& input,
                        std::vector<DenseTensor*> output,
                        DenseTensor* concated_out) {
  PADDLE_ENFORCE_GT(
      input.size(),
      static_cast<size_t>(0),
      errors::InvalidArgument("The ConcatTensor operator has no input."));
  PADDLE_ENFORCE_EQ(
      input.size(),
      output.size(),
      errors::InvalidArgument("The number of ConcatTensor operator's input and "
                              "output is not match, "
                              "input number is %u, output number is %u.",
                              input.size(),
                              output.size()));

  bool is_concated = IsConcated<T>(input);
  if (is_concated) {
    auto axis_val = phi::funcs::ComputeAxis(-1, input[0]->dims().size());
    std::vector<phi::DDim> input_dims;
    for (size_t i = 0; i < input.size(); ++i) {
      input_dims.push_back(input[i]->dims());
    }
    phi::DDim concated_out_dims =
        phi::funcs::ComputeAndCheckShape(true, input_dims, axis_val);
    concated_out->ShareDataWith(*input[0]);
    concated_out->Resize(concated_out_dims);
  } else {
    ConcatKernel<T, Context>(dev_ctx, input, -1, concated_out);
    for (size_t i = 0; i < input.size(); ++i) {
      auto input_ptr = const_cast<DenseTensor*>(input[i]);
      input_ptr->clear();
    }
  }

  auto concated_out_numel = concated_out->numel();
  auto concated_out_dims = concated_out->dims();
  concated_out->Resize({concated_out_numel});

  size_t offset = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    size_t len = static_cast<size_t>(output[i]->numel());
    auto dim = output[i]->dims();
    output[i]
        ->ShareDataWith(concated_out->Slice(
            static_cast<int64_t>(offset),
            static_cast<int64_t>(offset) + static_cast<int64_t>(len)))
        .Resize(dim);
    offset += len;
  }
  concated_out->Resize(concated_out_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(concat_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ConcatTensorKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
