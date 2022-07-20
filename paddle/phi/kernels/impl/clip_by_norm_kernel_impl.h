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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void ClipByNormFunctor(const Context& dev_ctx,
                       const DenseTensor& in,
                       float max_norm,
                       DenseTensor* output) {
  auto input = &in;
  dev_ctx.template Alloc<T>(output);

  PADDLE_ENFORCE_NOT_NULL(input,
                          phi::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));

  auto x = phi::EigenVector<T>::Flatten(*input);
  auto out = phi::EigenVector<T>::Flatten(*output);
  auto x_norm = x.square().sum().sqrt();
  auto* place = dev_ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<T>();
  auto epsilon = ((x_norm <= static_cast<T>(1e-30)).all().template cast<T>()) *
                 static_cast<T>(1e-6);

  auto scaling =
      temp + (static_cast<T>(1) - temp) * max_norm / (x_norm + epsilon);
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(input->numel());
  if (dev_ctx.GetPlace() == phi::CPUPlace()) {
    out.device(*place) = x * scaling.reshape(one_dim).eval().broadcast(m_dsize);
  } else {
    out.device(*place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
  }
}

}  // namespace phi
