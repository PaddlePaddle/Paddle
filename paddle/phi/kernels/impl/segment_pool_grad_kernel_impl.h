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

#pragma once

#include <string>

#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/segment_pooling.h"

namespace phi {

template <typename T, typename Context>
void SegmentPoolGradKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& segment_ids,
                           const DenseTensor& out,
                           const paddle::optional<DenseTensor>& summed_ids,
                           const DenseTensor& out_grad,
                           const std::string& pooltype,
                           DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, x_grad, static_cast<T>(0));

  auto index_type = segment_ids.type();
  if (index_type == DataType::INT32) {
    phi::funcs::SegmentPoolGradFunctor<Context, T, int> pool;
    pool(dev_ctx, x, out, out_grad, segment_ids, x_grad, summed_ids, pooltype);
  } else if (index_type == DataType::INT64) {
    phi::funcs::SegmentPoolGradFunctor<Context, T, int64_t> pool;
    pool(dev_ctx, x, out, out_grad, segment_ids, x_grad, summed_ids, pooltype);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported index type, Expected int, int64, but got %s.",
        index_type));
  }
}
}  // namespace phi
