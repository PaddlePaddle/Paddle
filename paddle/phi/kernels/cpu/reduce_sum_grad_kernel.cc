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

#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"
namespace phi {

template <typename T, typename Context>
void ComputeFromInput(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& input2,
                      const std::vector<int64_t>& dims,
                      DenseTensor* x_grad) {
  auto* input0 = &x;
  auto* output = x_grad;
  dev_ctx.template Alloc<T>(output);

  const auto* input2_d = input2.data<T>();
  auto* output_d = output->data<T>();

  // handle reduce_all
  if (input2.dims().size() == 1 && input2.dims()[0] == 1) {
    for (int64_t i = 0; i < phi::product(input0->dims()); ++i) {
      output_d[i] = input2_d[0];
    }
    return;
  }

  // handle reduce by one dimension
  int reduce_dim_index = dims[0];
  if (reduce_dim_index < 0) {
    reduce_dim_index += input0->dims().size();
  }

  auto& input_dim = input0->dims();
  int64_t before_dim = 1;
  for (int i = 0; i < reduce_dim_index; ++i) {
    before_dim *= input_dim[i];
  }
  int64_t reduce_dim = input_dim[reduce_dim_index];
  int64_t after_dim = 1;
  for (int i = reduce_dim_index + 1; i < input_dim.size(); ++i) {
    after_dim *= input_dim[i];
  }
  for (int64_t i = 0; i < before_dim; ++i) {
    for (int64_t j = 0; j < reduce_dim; ++j) {
      for (int64_t k = 0; k < after_dim; ++k) {
        output_d[i * reduce_dim * after_dim + j * after_dim + k] =
            input2_d[i * after_dim + k];
      }
    }
  }
}

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  if (dims.size() == 1) {
    if (out_grad.dtype() != x.dtype()) {
      DenseTensorMeta x_grad_meta(
          out_grad.dtype(), x_grad->dims(), x_grad->layout());
      DenseTensor x_grad_tmp =
          phi::Empty<Context>(dev_ctx, std::move(x_grad_meta));

      ComputeFromInput<T, Context>(
          dev_ctx, x, out_grad, dims.GetData(), &x_grad_tmp);

      phi::CastKernel<T>(dev_ctx, x_grad_tmp, x.dtype(), x_grad);

    } else {
      ComputeFromInput<T, Context>(
          dev_ctx, x, out_grad, dims.GetData(), x_grad);
    }
  }

  ReduceGradKernel<Context, T, funcs::SumGradFunctor, true>(dev_ctx,
                                                            x,
                                                            paddle::none,
                                                            out_grad,
                                                            dims.GetData(),
                                                            keep_dim,
                                                            reduce_all,
                                                            x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
