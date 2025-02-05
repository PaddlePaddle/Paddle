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

#include "paddle/phi/kernels/scatter_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ScatterKernel(const Context &ctx,
                   const DenseTensor &x,
                   const DenseTensor &index,
                   const DenseTensor &updates,
                   bool overwrite,
                   DenseTensor *out) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;
  out->Resize(x.dims());
  auto *x_data = reinterpret_cast<const XPUTypeT *>(x.data<T>());
  auto *updates_data = reinterpret_cast<const XPUTypeT *>(updates.data<T>());
  auto *out_data = reinterpret_cast<XPUTypeT *>(ctx.template Alloc<T>(out));
  int ret = xpu::copy(ctx.x_context(), x_data, out_data, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
  // Apply ScatterUpdate: Out[index] = Updates[:]
  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  // check index of shape 1-D
  PADDLE_ENFORCE_EQ(
      index.dims().size() == 1 || index.dims().size() == 0 ||
          (index.dims().size() == 2 && index.dims()[1] == 1),
      true,
      common::errors::InvalidArgument(
          "index's shape is error, "
          "expect index'dims shape is 0, 1, 2 (index.dims[1] should "
          "be 1), 0 but got index'dims shape is %d",
          index.dims().size()));

  int index_size =
      static_cast<int>(index.dims().size() == 0 ? 1 : index.dims()[0]);
  auto x_dims = x.dims();
  auto update_dims = updates.dims();
  if (index.dims().size() != 0) {
    // only check when the updates tensor is not a 0D tensor
    for (int i = 1; i < x_dims.size(); i++)
      PADDLE_ENFORCE_EQ(
          x_dims[i],
          update_dims[i],
          common::errors::InvalidArgument(
              "The dimensions of the source tensor and target tensor should"
              " match, but received source tensor's %d-th dimension is %d,"
              "target tensor's %d-th dimension is %d.",
              i,
              x_dims[i],
              i,
              update_dims[i]));
  }

  int dim0 = static_cast<int>(x.dims()[0]);
  int dim1 = static_cast<int>(
      common::product(common::slice_ddim(x_dims, 1, x_dims.size())));

  DenseTensor indices_cpu(index.type());
  phi::Copy(ctx, index, phi::CPUPlace(), true, &indices_cpu);

  int r = 0;
  if (index_type == phi::DataType::INT32) {
    auto index_data = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> indices{
        indices_cpu.data<int>(), index_size, index_data};
    r = xpu::scatter(ctx.x_context(),
                     updates_data,
                     out_data,
                     indices,
                     dim0,
                     dim1,
                     overwrite);
  } else {
    auto index_data = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> indices{
        indices_cpu.data<int64_t>(), index_size, index_data};
    r = xpu::scatter(ctx.x_context(),
                     updates_data,
                     out_data,
                     indices,
                     dim0,
                     dim1,
                     overwrite);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
}

}  // namespace phi

PD_REGISTER_KERNEL(scatter,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScatterKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
