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

#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const DenseTensor*>& x,
                  const Scalar& axis_scalar,
                  DenseTensor* out) {
  int64_t axis = axis_scalar.to<int64_t>();

  if (UNLIKELY(x[0]->dims().size() == 0)) {
    // for dims is 0 specially
    phi::DDim tmp_1dim, out_dims;
    out_dims[0] = x.size();
    tmp_1dim[0] = 1;

    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);

    size_t output_offset = 0;
    for (auto* in : x) {
      if (in->numel() == 0UL) {
        continue;
      }
      auto in_stride = phi::stride_numel(tmp_1dim);
      auto out_stride = phi::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          dev_ctx,
          axis,
          out->data<T>() + output_offset,
          out_stride,
          in->data<T>(),
          in_stride,
          in_stride[axis]);
      output_offset += in_stride[axis];
    }
    return;
  }

  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());

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
            phi::errors::Unimplemented(
                "The lod level of all input LoDTensors should be same. "
                "Maybe different lod level of input LoDTensors can concat,"
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
        auto in_lod = phi::ConvertToLengthBasedLoD(x[i]->lod());
        phi::AppendLoD(out_lod, in_lod);
      }
    }
  }

  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && x.size() < 10) {
    size_t output_offset = 0;
    for (auto* in : x) {
      if (in->numel() == 0UL) {
        continue;
      }
      auto in_stride = phi::stride_numel(in->dims());
      auto out_stride = phi::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          dev_ctx,
          axis,
          out->data<T>() + output_offset,
          out_stride,
          in->data<T>(),
          in_stride,
          in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    std::vector<phi::DenseTensor> inputs;
    for (size_t j = 0; j < x.size(); ++j) {
      if (x[j]->numel() > 0) {
        inputs.push_back(*x[j]);
      } else {
        continue;
      }
    }
    phi::funcs::ConcatFunctor<Context, T> functor;
    functor(dev_ctx, inputs, axis, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(concat,
                   GPU,
                   ALL_LAYOUT,
                   phi::ConcatKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
