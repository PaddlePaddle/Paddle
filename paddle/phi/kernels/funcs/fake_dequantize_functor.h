/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
class DequantizeFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor* in,
                  const DenseTensor* scale,
                  T max_range,
                  DenseTensor* out);
};

template <typename Context, typename T>
class ChannelDequantizeFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor* in,
                  const DenseTensor** scales,
                  const int scale_num,
                  T max_range,
                  const int quant_axis,
                  const int x_num_col_dims,
                  DenseTensor* out);
};

}  // namespace funcs
}  // namespace phi
