/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {
// template struct ColwiseSum<phi::GPUContext, double>;
// The ColwiseSum<phi::GPUContext, double> failed in debug
// mode,
// and only failed for this case. So reimplemented it.
template <>
void ColwiseSum<phi::GPUContext, double>::operator()(
    const phi::GPUContext& dev_ctx,
    const phi::DenseTensor& input,
    phi::DenseTensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(),
                    size,
                    common::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor column"
                        " dimension. Expected vector size=%d, but received %d",
                        size,
                        vector->numel()));
  phi::DenseTensor one;
  one.Resize({in_dims[0]});
  dev_ctx.template Alloc<double>(&one);

  SetConstant<phi::GPUContext, double> set;
  set(dev_ctx, &one, static_cast<double>(1.0));
  phi::funcs::GetBlas<phi::GPUContext, double>(dev_ctx).GEMV(
      true,
      static_cast<int>(in_dims[0]),
      static_cast<int>(in_dims[1]),
      1.0,
      input.data<double>(),
      one.data<double>(),
      0.0,
      vector->data<double>());
}

// template struct RowwiseSum<phi::GPUContext, double>;
// TODO(zcd): Following ColwiseSum format, need to confirm.
// The RowwiseSum<phi::GPUContext, double> failed in debug
// mode,
template <>
void RowwiseSum<phi::GPUContext, double>::operator()(
    const phi::GPUContext& dev_ctx,
    const phi::DenseTensor& input,
    phi::DenseTensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(),
                    in_dims[0],
                    common::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor row"
                        " dimension. Expected vector size=%d, but received %d",
                        in_dims[0],
                        vector->numel()));
  phi::DenseTensor one;
  one.Resize({size});
  dev_ctx.template Alloc<double>(&one);

  SetConstant<phi::GPUContext, double> set;
  set(dev_ctx, &one, static_cast<double>(1.0));
  phi::funcs::GetBlas<phi::GPUContext, double>(dev_ctx).GEMV(
      true,
      static_cast<int>(in_dims[1]),
      static_cast<int>(in_dims[0]),
      1.0,
      one.data<double>(),
      input.data<double>(),
      0.0,
      vector->data<double>());
}

}  // namespace funcs
}  // namespace phi
