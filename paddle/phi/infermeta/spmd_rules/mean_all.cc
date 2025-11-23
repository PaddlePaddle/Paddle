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

#include "paddle/phi/infermeta/spmd_rules/mean_all.h"
#include "glog/logging.h"

namespace phi {
namespace distributed {

SpmdInfo MeanAllInferSpmd(const DistMetaTensor& x) {
  VLOG(4) << "MeanAllInferSpmd Call ReductionInferSpmdBase";
  return ReductionInferSpmdBase(
      x, {}, false, static_cast<int>(ReduceType::kRedAvg));
}

SpmdInfo MeanAllGradInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& out_grad) {
  VLOG(4) << "MeanAllGradInferSpmd Call ReductionGradInferSpmd";
  return ReductionGradInferSpmd(x, out_grad, {}, false, true);
}

}  // namespace distributed
}  // namespace phi
