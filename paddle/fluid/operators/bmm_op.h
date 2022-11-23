/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#ifndef PADDLE_FLUID_OPERATORS_BMM_OP_H_
#define PADDLE_FLUID_OPERATORS_BMM_OP_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

static void ReshapeTensorIntoMatrixSequence(
    phi::DenseTensor *x, const phi::funcs::MatDescriptor &descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }

  x->Resize({descriptor.batch_size_, h, w});
}

static void ReshapeXYOutIntoMatrixSequence(phi::DenseTensor *x,
                                           phi::DenseTensor *y,
                                           phi::DenseTensor *out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = x->dims();
  auto y_dim = y->dims();
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, false);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, false);

  out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
               mat_dim_x.height_,
               mat_dim_y.width_});

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_FLUID_OPERATORS_BMM_OP_H_
