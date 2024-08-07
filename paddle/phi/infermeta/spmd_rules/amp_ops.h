// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {
SpmdInfo CheckFiniteAndUnscaleSpmd(const std::vector<DistMetaTensor>& xs,
                                   const DistMetaTensor& scale);

SpmdInfo UpdateLossScalingSpmd(const std::vector<DistMetaTensor>& xs,
                               const DistMetaTensor& found_infinite,
                               const DistMetaTensor& prev_loss_scaling,
                               const DistMetaTensor& in_good_steps,
                               const DistMetaTensor& in_bad_steps,
                               int incr_every_n_steps,
                               int decr_every_n_nan_or_inf,
                               float incr_ratio,
                               float decr_ratio,
                               Scalar stop_update = false);
}  // namespace distributed
}  // namespace phi
