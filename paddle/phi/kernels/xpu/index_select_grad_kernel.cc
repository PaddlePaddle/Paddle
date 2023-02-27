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
void IndexSelectGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int dim,
                           DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  XPUType* in_grad_data =
      reinterpret_cast<XPUType*>(ctx.template Alloc<T>(x_grad));
  auto* out_grad_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());

  auto input_dim = x_grad->dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  auto output_dim = out_grad.dims();
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

  std::vector<int> x_grad_shape = phi::vectorize<int>(input_dim);
  std::vector<int> out_grad_shape = phi::vectorize<int>(output_dim);
  int r = 0;
  if (index_type == phi::DataType::INT64) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "index_select_grad do not support index int64_t"));
  } else {
    const int* index_data = index.data<int>();
#if 1
    r = xpu::index_select_grad<XPUType>(ctx.x_context(),
                                        nullptr,
                                        index_data,
                                        out_grad_data,
                                        dim,
                                        in_grad_data,
                                        out_grad_shape,
                                        x_grad_shape);
#else
    (void)(index_data);
    (void)(out_grad_data);
    (void)(dim);
    (void)(in_grad_data);
    (void)(out_grad_shape);
    (void)(x_grad_shape);
#endif
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_select_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    index_select_grad, XPU, ALL_LAYOUT, phi::IndexSelectGradKernel, float) {}
