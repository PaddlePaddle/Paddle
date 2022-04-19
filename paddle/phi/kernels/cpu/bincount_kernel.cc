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

#include "paddle/phi/kernels/bincount_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename Context, typename T, typename InputT>
void BincountInner(const Context& dev_ctx,
                   const DenseTensor& x,
                   const paddle::optional<const DenseTensor&> weights,
                   int minlength,
                   DenseTensor* out) {
  const DenseTensor* input = &x;
  DenseTensor* output = out;
  const InputT* input_data = input->data<InputT>();

  auto input_numel = input->numel();

  if (input_data == nullptr) {
    phi::DDim out_dim{0};
    output->Resize(out_dim);
    dev_ctx.template Alloc<InputT>(output);
    return;
  }

  PADDLE_ENFORCE_GE(
      *std::min_element(input_data, input_data + input_numel),
      static_cast<InputT>(0),
      phi::errors::InvalidArgument(
          "The elements in input tensor must be non-negative ints"));

  int64_t output_size = static_cast<int64_t>(*std::max_element(
                            input_data, input_data + input_numel)) +
                        1L;
  output_size = std::max(output_size, static_cast<int64_t>(minlength));

  phi::DDim out_dim{output_size};
  output->Resize(out_dim);

  bool has_weights = weights.is_initialized();

  if (has_weights) {
    const T* weights_data = weights->data<T>();
    if (weights->dtype() == DataType::FLOAT32) {
      float* output_data = dev_ctx.template Alloc<float>(output);
      phi::funcs::SetConstant<Context, float>()(
          dev_ctx, output, static_cast<float>(0));
      for (int64_t i = 0; i < input_numel; i++) {
        output_data[input_data[i]] += static_cast<float>(weights_data[i]);
      }
    } else {
      double* output_data = dev_ctx.template Alloc<double>(output);
      phi::funcs::SetConstant<Context, double>()(
          dev_ctx, output, static_cast<double>(0));
      for (int64_t i = 0; i < input_numel; i++) {
        output_data[input_data[i]] += static_cast<double>(weights_data[i]);
      }
    }

  } else {
    int64_t* output_data = dev_ctx.template Alloc<int64_t>(output);
    phi::funcs::SetConstant<Context, int64_t>()(dev_ctx, output, 0L);
    for (int64_t i = 0; i < input_numel; i++) {
      output_data[input_data[i]] += 1L;
    }
  }
}

template <typename T, typename Context>
void BincountKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const paddle::optional<const DenseTensor&> weights,
                    int minlength,
                    DenseTensor* out) {
  if (x.dtype() == DataType::INT32) {
    BincountInner<Context, T, int>(dev_ctx, x, weights, minlength, out);
  } else if (x.dtype() == DataType::INT64) {
    BincountInner<Context, T, int64_t>(dev_ctx, x, weights, minlength, out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(bincount,
                   CPU,
                   ALL_LAYOUT,
                   phi::BincountKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
