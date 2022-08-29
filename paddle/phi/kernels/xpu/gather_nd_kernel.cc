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

#include "paddle/phi/kernels/gather_nd_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherNdKernel(const Context &ctx,
                    const DenseTensor &x,
                    const DenseTensor &index,
                    DenseTensor *out) {
  ctx.template Alloc<T>(out);
  const auto &index_type = index.dtype();

  if (x.numel() == 0) return;

  if (index.numel() == 0) {
    phi::Copy(ctx, x, phi::XPUPlace(), true, out);
    return;
  }

  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   DataType::INT32,
                                   DataType::INT64));

  auto x_shape = phi::vectorize<int>(x.dims());
  auto index_shape = phi::vectorize<int>(index.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int> x_vec = {
      x_shape.data(), static_cast<int>(x_shape.size()), nullptr};

  int ret = XPU_SUCCESS;
  if (index_type == DataType::INT32) {
    ret = xpu::gather_nd<T, int>(ctx.x_context(),
                                 x.data<T>(),
                                 index.data<int>(),
                                 out->data<T>(),
                                 x_vec,
                                 index_shape);
  } else {
    ret = xpu::gather_nd<T, int64_t>(ctx.x_context(),
                                     x.data<T>(),
                                     index.data<int64_t>(),
                                     out->data<T>(),
                                     x_vec,
                                     index_shape);
  }
  PADDLE_ENFORCE_EQ(
      ret,
      XPU_SUCCESS,
      phi::errors::External("XPU gather_nd kernel return wrong value[%d %s]",
                            ret,
                            XPUAPIErrorMsg[ret]));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gather_nd, XPU, ALL_LAYOUT, phi::GatherNdKernel, float, int64_t, int) {}
