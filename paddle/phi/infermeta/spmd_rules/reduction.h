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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo ReductionInferSpmd(const DistMetaTensor& x,
                            const std::vector<int64_t>& axis,
                            bool keep_dim);

SpmdInfo ReductionInferSpmdBase(const DistMetaTensor& x,
                                const std::vector<int64_t>& axis,
                                bool keep_dim,
                                int reduce_type);

// This infer spmd function only use in dynamic mode for it uses
// IntArray as parameter. The IntArray may contain vector of tensor
// which is not support in static mode. So we separate these two and
// use dynamic infer_spmd invoke static infer_spmd function.
SpmdInfo ReductionMeanInferSpmdDynamic(const DistMetaTensor& x,
                                       const IntArray& axis,
                                       bool keep_dim);

SpmdInfo ReductionSumInferSpmdDynamic(const DistMetaTensor& x,
                                      const IntArray& axis,
                                      DataType dtype,
                                      bool keep_dim);

SpmdInfo ReductionMaxInferSpmdDynamic(const DistMetaTensor& x,
                                      const IntArray& axis,
                                      bool keep_dim);

SpmdInfo ReductionAllInferSpmdDynamic(const DistMetaTensor& x,
                                      const IntArray& axis,
                                      bool keep_dim);

SpmdInfo ReductionInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int64_t>& axis,
                                   bool keep_dim);

SpmdInfo ReductionGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& out_grad,
                                const IntArray& axis,
                                bool keep_dim,
                                bool reduce_all);

SpmdInfo ReductionGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& out,
                                const DistMetaTensor& out_grad,
                                const IntArray& axis,
                                bool keep_dim,
                                bool reduce_all);

}  // namespace distributed
}  // namespace phi
