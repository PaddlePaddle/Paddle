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

#include <vector>

#include "paddle/fluid/operators/math/sequence_padding.h"
#include "paddle/fluid/operators/math/sequence_scale.h"
#include "paddle/phi/backends/dynload/warpctc.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void WarpctcGradKernel(const Context& dev_ctx,
                       const DenseTensor& warpctc_grad,
                       const DenseTensor& logits,
                       const DenseTensor& loss_grad,
                       const paddle::optional<const DenseTensor&> logits_length,
                       int blank,
                       bool norm_by_times,
                       DenseTensor* logits_grad) {
  dev_ctx.template Alloc<T>(logits_grad);

  if (logits_length.is_initialized()) {
    int max_seq_length = warpctc_grad.dims()[0];  // Tmax
    int num_sequences = warpctc_grad.dims()[1];   // B
    int seq_width = warpctc_grad.dims()[2];       // D

    // B
    auto logits_len_e = EigenTensor<int64_t, 1>::From(*logits_length);
    // (B, 1)
    auto loss_grad_e = EigenTensor<T, 2>::From(loss_grad);
    // (T, B, D)
    auto warpctc_grad_e = EigenTensor<T, 3>::From(warpctc_grad);

    auto logits_grad_e = EigenTensor<T, 3>::From(*logits_grad);

    Eigen::DSizes<int, 3> grad_shape(1, num_sequences, 1);
    Eigen::DSizes<int, 3> bcast(max_seq_length, 1, seq_width);
    auto logits_g = warpctc_grad_e *
                    loss_grad_e.reshape(grad_shape).broadcast(bcast).eval();

    auto* place = dev_ctx.eigen_device();
    if (norm_by_times) {
      auto scales = logits_len_e.cast<T>()
                        .inverse()
                        .reshape(grad_shape)
                        .broadcast(bcast)
                        .eval();
      logits_grad_e.device(*place) = logits_g * scales;
    } else {
      logits_grad_e.device(*place) = logits_g;
    }
  } else {
    paddle::operators::math::UnpaddingLoDTensorFunctor<Context, T>()(
        dev_ctx,
        warpctc_grad,
        logits_grad,
        -1,
        0,
        norm_by_times,
        paddle::operators::math::kLengthBatchWidth);

    const T* loss_grad_data = loss_grad.data<T>();
    paddle::operators::math::ScaleLoDTensorFunctor<Context, T>()(
        dev_ctx, loss_grad_data, logits_grad);
  }
}

}  // namespace phi
