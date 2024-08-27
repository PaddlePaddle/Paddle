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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {
SpmdInfo FusedDropoutAddSpmdBase(const DistMetaTensor& x,
                                 const DistMetaTensor& y);

SpmdInfo FusedDropoutAddSpmdReverseBase(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out,
                                        const DistMetaTensor& seed_offset);

SpmdInfo FusedDropoutAddGradInferSpmdBase(const DistMetaTensor& seed_offset,
                                          const DistMetaTensor& out_grad);

SpmdInfo FusedDropoutAddSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& seed_tensor,
                             const Scalar& p,
                             bool is_test,
                             const std::string& mode,
                             int seed,
                             bool fix_seed);

SpmdInfo FusedDropoutAddSpmdReverse(const DistMetaTensor& x,
                                    const DistMetaTensor& y,
                                    const DistMetaTensor& seed_tensor,
                                    const DistMetaTensor& out,
                                    const DistMetaTensor& seed_offset,
                                    const Scalar& p,
                                    bool is_test,
                                    const std::string& mode,
                                    int seed,
                                    bool fix_seed);

SpmdInfo FusedDropoutAddGradInferSpmd(const DistMetaTensor& seed_offset,
                                      const DistMetaTensor& out_grad,
                                      const Scalar& p,
                                      bool is_test,
                                      std::string mode,
                                      bool fix_seed);
}  // namespace distributed
}  // namespace phi
