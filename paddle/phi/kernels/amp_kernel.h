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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite);

template <typename T, typename Context>
void UpdateLossScalingKernel(const Context& dev_ctx,
                             const std::vector<const DenseTensor*>& xs,
                             const DenseTensor& found_infinite,
                             const DenseTensor& prev_loss_scaling,
                             const DenseTensor& in_good_steps,
                             const DenseTensor& in_bad_steps,
                             int incr_every_n_steps,
                             int decr_every_n_nan_or_inf,
                             float incr_ratio,
                             float decr_ratio,
                             const Scalar& stop_update,
                             std::vector<DenseTensor*> outs,
                             DenseTensor* loss_scaling,
                             DenseTensor* out_good_steps,
                             DenseTensor* out_bad_steps);

}  // namespace phi
