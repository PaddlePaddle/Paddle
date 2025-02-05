/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo AdamInferSpmdDynamic(
    const DistMetaTensor& param,
    const DistMetaTensor& grad,
    const DistMetaTensor& learning_rate,
    const DistMetaTensor& moment1,
    const DistMetaTensor& moment2,
    const paddle::optional<DistMetaTensor>& moment2_max,
    const DistMetaTensor& beta1_pow,
    const DistMetaTensor& beta2_pow,
    const DistMetaTensor& master_param,
    const DistMetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad);

SpmdInfo AdamwInferSpmdDynamic(
    const DistMetaTensor& param,
    const DistMetaTensor& grad,
    const DistMetaTensor& learning_rate,
    const DistMetaTensor& moment1,
    const DistMetaTensor& moment2,
    const paddle::optional<DistMetaTensor>& moment2_max,
    const DistMetaTensor& beta1_pow,
    const DistMetaTensor& beta2_pow,
    const DistMetaTensor& master_param,
    const DistMetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    float lr_ratio,
    float coeff,
    bool with_decay,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad);

SpmdInfo SgdInferSpmd(const DistMetaTensor& param,
                      const DistMetaTensor& learning_rate,
                      const DistMetaTensor& grad,
                      const DistMetaTensor& master_param,
                      bool multi_precision);

}  // namespace distributed
}  // namespace phi
