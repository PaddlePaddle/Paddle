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

#include "paddle/phi/infermeta/spmd_rules/logsumexp.h"
#include "glog/logging.h"

namespace phi {
namespace distributed {

SpmdInfo LogSumExpInferSpmd(const DistMetaTensor& x,
                            const std::vector<int>& axis,
                            bool keepdims,
                            bool reduce_all) {
  VLOG(4) << "LogSumExpInferSpmd Call ReductionInferSpmd";
  std::vector<int64_t> new_axis(axis.begin(), axis.end());
  return ReductionInferSpmd(x, new_axis, keepdims);
}

SpmdInfo LogSumExpInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int>& axis,
                                   bool keepdims,
                                   bool reduce_all) {
  VLOG(4) << "LogSumExpInferSpmdReverse Call ReductionInferSpmdReverse";
  std::vector<int64_t> new_axis(axis.begin(), axis.end());
  return ReductionInferSpmdReverse(x, out, new_axis, keepdims);
}

SpmdInfo LogSumExpGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& out,
                                const DistMetaTensor& out_grad,
                                const std::vector<int>& axis,
                                bool keepdims,
                                bool reduce_all) {
  VLOG(4) << "LogSumExpGradInferSpmd Call ReductionGradInferSpmd";
  std::vector<int64_t> new_axis(axis.begin(), axis.end());
  return ReductionGradInferSpmd(
      x, out, out_grad, new_axis, keepdims, reduce_all);
}

}  // namespace distributed
}  // namespace phi
