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

#include "paddle/phi/kernels/index_select_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"

namespace phi {

template <typename T, typename Context>
void IndexSelectGradKernel(const Context& ctx,
                           const DenseTensor& x UNUSED,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int dim,
                           DenseTensor* x_grad) {
  if (dim < 0) {
    dim += out_grad.dims().size();
  }
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  if (index_type == phi::DataType::INT32) {
    IndexSelectGradInner<Context, T, int>(ctx, out_grad, index, x_grad, dim);
  } else if (index_type == phi::DataType::INT64) {
    IndexSelectGradInner<Context, T, int64_t>(
        ctx, out_grad, index, x_grad, dim);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexSelectGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   int,
                   int64_t) {}
