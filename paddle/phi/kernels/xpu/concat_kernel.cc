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

#include "paddle/phi/kernels/concat_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const DenseTensor*>& x,
                  const Scalar& axis_scalar,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t axis = axis_scalar.to<int64_t>();
  PADDLE_ENFORCE_NE(
      x[0],
      nullptr,
      common::errors::InvalidArgument("The input should not be null."));
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  PADDLE_ENFORCE_GE(
      axis,
      0,
      common::errors::InvalidArgument("concat: axis should be larger than or "
                                      "equal to 0, but received axis is %d.",
                                      axis));
  PADDLE_ENFORCE_LT(axis,
                    x[0]->dims().size(),
                    common::errors::InvalidArgument(
                        "concat: axis should be less than x[0]->dims()!"
                        "But received axis is %d, while x[0]->dims()"
                        "size is %d.",
                        axis,
                        x[0]->dims().size()));

  std::vector<phi::DDim> x_dims;
  for (size_t i = 0; i < x.size(); ++i) {
    x_dims.push_back(x[i]->dims());
  }

  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  // If axis is 0, the lod of the output is not the same as inputs.

  if (axis == 0 && x[0]->lod().size() > 0) {
    size_t lod_size_0 = x[0]->lod().size();
    size_t lod_size = lod_size_0;
    for (size_t i = 1; i < x.size(); ++i) {
      if (x[i]->lod().size() > 0) {
        PADDLE_ENFORCE_EQ(
            x[i]->lod().size(),
            lod_size_0,
            common::errors::Unimplemented(
                "The lod level of all input DenseTensors should be same. "
                "Maybe different lod level of input DenseTensors can concat,"
                "it is not supported currently. The lod level of %dth input "
                "is %d and first input is %d.",
                i,
                x[i]->lod().size(),
                lod_size_0));
      } else {
        lod_size = 0;
        break;
      }
    }
    if (lod_size) {
      auto* out_lod = out->mutable_lod();
      for (size_t i = 1; i < x.size(); ++i) {
        auto in_lod = phi::ConvertToLengthBasedLegacyLoD(x[i]->lod());
        phi::AppendLegacyLoD(out_lod, in_lod);
      }
    }
  }

  std::vector<std::vector<int>> xdims_list;
  std::vector<const XPUType*> ptrs;
  for (unsigned int i = 0; i < x.size(); ++i) {
    if (x[i] && x[i]->numel() > 0) {
      ptrs.push_back(reinterpret_cast<const XPUType*>(x[i]->data<T>()));
      int size = x[i]->dims().size();
      std::vector<int> tmp_dims(size);
      for (int j = 0; j < size; ++j) {
        tmp_dims[j] = x[i]->dims()[j];
      }
      xdims_list.push_back(tmp_dims);
    }
  }

  PADDLE_ENFORCE_GT(xdims_list.size(),
                    0,
                    common::errors::InvalidArgument("No tensor need concat"));

  int r = xpu::concat<XPUType>(dev_ctx.x_context(),
                               ptrs,
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               xdims_list,
                               axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat");
}

}  // namespace phi

PD_REGISTER_KERNEL(concat,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConcatKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t) {}
