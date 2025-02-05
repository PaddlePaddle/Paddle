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

#include "paddle/phi/kernels/index_put_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/xpu/index_put_xpu_utils.h"

namespace phi {
template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices_v,
                        const DenseTensor& value,
                        const DenseTensor& out_grad,
                        bool accumulate,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      common::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  // All bool indices are converted to integers currently
  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices_v, &tmp_args);

  if (int_indices_v.empty()) {
    if (x_grad) {
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    if (value_grad) {
      FullKernel<T, Context>(dev_ctx,
                             common::vectorize(value_grad->dims()),
                             0.0f,
                             value_grad->dtype(),
                             value_grad);
    }
    return;
  }

  auto bd_dims = funcs::BroadCastTensorsDims(int_indices_v);
  DenseTensor res_indices(DataType::INT64);
  // Broadcast and merge indices
  XPUDealWithIndices<Context>(dev_ctx, int_indices_v, bd_dims, &res_indices);
  auto index_shape = common::vectorize<int64_t>(res_indices.dims());
  xpu::VectorParam<int64_t> index_param = {
      nullptr, res_indices.numel(), res_indices.data<int64_t>()};
  auto xshape = common::vectorize<int64_t>(x.dims());
  xpu::VectorParam<int64_t> xshape_param = {
      xshape.data(), static_cast<int64_t>(xshape.size()), nullptr};

  int64_t value_rank = bd_dims.size() + (xshape.size() - int_indices_v.size());
  std::vector<int64_t> value_shape_bd(value_rank);
  std::copy(index_shape.begin(), index_shape.end() - 1, value_shape_bd.begin());
  std::copy(xshape.begin() + int_indices_v.size(),
            xshape.end(),
            value_shape_bd.begin() + index_shape.size() - 1);
  int ret = xpu::SUCCESS;
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    if (!accumulate) {
      DenseTensor zero_tensor(x_grad->dtype());
      FullKernel<T, Context>(
          dev_ctx, value_shape_bd, 0.0f, zero_tensor.dtype(), &zero_tensor);
      ret = xpu::scatter_nd<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(x_grad->data<T>()),
          reinterpret_cast<const XPUType*>(zero_tensor.data<T>()),
          reinterpret_cast<XPUType*>(x_grad->data<T>()),
          index_param,
          xshape_param,
          index_shape,
          false);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "scatter_nd");
    }
  }
  if (value_grad) {
    auto value_shape = common::vectorize<int64_t>(value_grad->dims());
    dev_ctx.template Alloc<T>(value_grad);
    if (value_shape != value_shape_bd) {
      std::vector<int64_t> compress_dims;
      std::vector<int64_t> dims_without_1;
      funcs::CalCompressedDimsWith1AndWithout1(
          &value_shape_bd, &value_shape, &compress_dims, &dims_without_1);
      DenseTensor value_grad_bd(value_grad->dtype());
      value_grad_bd.Resize(common::make_ddim(value_shape_bd));
      dev_ctx.template Alloc<T>(&value_grad_bd);
      ret = xpu::gather_nd<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(out_grad.data<T>()),
          res_indices.data<int64_t>(),
          reinterpret_cast<XPUType*>(value_grad_bd.data<T>()),
          xshape_param,
          index_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather_nd");
      IntArray v_axis(compress_dims);
      auto pre_dims = value_grad->dims();
      SumKernel<T>(dev_ctx,
                   value_grad_bd,
                   v_axis,
                   value_grad->dtype(),
                   false,
                   value_grad);
      value_grad->Resize(pre_dims);
    } else {
      ret = xpu::gather_nd<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(out_grad.data<T>()),
          res_indices.data<int64_t>(),
          reinterpret_cast<XPUType*>(value_grad->data<T>()),
          xshape_param,
          index_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather_nd");
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexPutGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
