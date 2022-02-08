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

#include "paddle/pten/kernels/concat_kernel.h"

#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/lod_utils.h"
#include "paddle/pten/kernels/funcs/concat_funcs.h"
#include "paddle/pten/kernels/gpu/concat_and_split.h"

namespace pten {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<DenseTensor>& x,
                  const Scalar& axis_scalar,
                  DenseTensor* out) {
  int64_t axis = axis_scalar.to<int64_t>();

  axis = pten::funcs::ComputeAxis(axis, x[0].dims().size());

  std::vector<pten::DDim> x_dims;
  for (size_t i = 0; i < x.size(); ++i) {
    x_dims.push_back(x[i].dims());
  }

  pten::DDim out_dims = pten::funcs::ComputeAndCheckShape(true, x_dims, axis);
  out->Resize(out_dims);
  out->mutable_data<T>(dev_ctx.GetPlace());

  // If axis is 0, the lod of the output is not the same as inputs.
  if (axis == 0 && x[0].lod().size() > 0) {
    size_t lod_size_0 = x[0].lod().size();
    size_t lod_size = lod_size_0;
    for (size_t i = 1; i < x.size(); ++i) {
      if (x[i].lod().size() > 0) {
        PADDLE_ENFORCE_EQ(
            x[i].lod().size(),
            lod_size_0,
            paddle::platform::errors::Unimplemented(
                "The lod level of all input LoDTensors should be same. "
                "Maybe different lod level of input LoDTensors can concat,"
                "it is not supported currently. The lod level of %dth input "
                "is %d and first input is %d.",
                i,
                x[i].lod().size(),
                lod_size_0));
      } else {
        lod_size = 0;
        break;
      }
    }
    if (lod_size) {
      auto* out_lod = out->mutable_lod();
      for (size_t i = 1; i < x.size(); ++i) {
        auto in_lod = pten::ConvertToLengthBasedLoD(x[i].lod());
        pten::AppendLoD(out_lod, in_lod);
      }
    }
  }

  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && x.size() < 10) {
    size_t output_offset = 0;
    for (auto& in : x) {
      if (in.numel() == 0UL) {
        continue;
      }
      auto in_stride = paddle::framework::stride_numel(in.dims());
      auto out_stride = paddle::framework::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          dev_ctx,
          axis,
          out->data<T>() + output_offset,
          out_stride,
          in.data<T>(),
          in_stride,
          in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    std::vector<pten::DenseTensor> inputs;
    for (size_t j = 0; j < x.size(); ++j) {
      if (x[j].numel() > 0) {
        inputs.push_back(x[j]);
      } else {
        continue;
      }
    }
    ConcatImpl<T, Context>(dev_ctx, inputs, axis, out);
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(concat,
                   GPU,
                   ALL_LAYOUT,
                   pten::ConcatKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   paddle::platform::float16,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
