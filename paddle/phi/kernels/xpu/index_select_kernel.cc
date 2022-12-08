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

#include "paddle/phi/kernels/index_select_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename T, typename Context>
void IndexSelectKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       int dim,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto* in_data = x.data<T>();
  std::vector<int> in_shape = phi::vectorize<int>(input_dim);
  int index_len = output->dims()[dim];
  T* out_data = ctx.template Alloc<T>(output);
  int r = 0;
  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    r = xpu::gather<T, int64_t>(ctx.x_context(),
                                in_data,
                                index_data,
                                out_data,
                                in_shape,
                                index_len,
                                dim);
  } else {
    const int* index_data = index.data<int>();
    r = xpu::gather<T, int>(ctx.x_context(),
                            in_data,
                            index_data,
                            out_data,
                            in_shape,
                            index_len,
                            dim);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexSelectKernel,
                   float,
                   int,
                   int64_t) {}
