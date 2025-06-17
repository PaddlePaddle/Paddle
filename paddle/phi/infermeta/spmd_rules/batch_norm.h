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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {
SpmdInfo BatchNormInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& mean,
                            const DistMetaTensor& variance,
                            const DistMetaTensor& scale,
                            const DistMetaTensor& bias,
                            const bool is_test = false,
                            const float momentum = 0.9,
                            const float epsilon = 1e-05,
                            const std::string& data_format = "NCHW",
                            const bool use_global_stats = false,
                            const bool trainable_statistics = false);
SpmdInfo BatchNormInferSpmdStatic(const DistMetaTensor& x,
                                  const DistMetaTensor& mean,
                                  const DistMetaTensor& variance,
                                  const DistMetaTensor& scale,
                                  const DistMetaTensor& bias);

SpmdInfo BatchNormGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& scale,
                                const DistMetaTensor& bias,
                                const DistMetaTensor& mean_out,
                                const DistMetaTensor& variance_out,
                                const DistMetaTensor& saved_mean,
                                const DistMetaTensor& saved_variance,
                                const DistMetaTensor& reserve_space,
                                const DistMetaTensor& out_grad,
                                const float momentum = 0.9,
                                const float epsilon = 1e-05,
                                const std::string& data_format = "NCHW",
                                const bool is_test = false,
                                const bool use_global_stats = false,
                                const bool trainable_statistics = false);

}  // namespace distributed
}  // namespace phi
