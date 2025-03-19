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

#include "paddle/phi/infermeta/spmd_rules/squared_l2_norm.h"
#include "paddle/phi/infermeta/spmd_rules/reduction.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo SquaredL2NormInferSpmd(const DistMetaTensor& x) {
  VLOG(4) << "SquaredL2NormInferSpmd:";
  VLOG(4) << "Using ReductionInferSpmd Rule as internal implement.";
  SpmdInfo info = ReductionInferSpmdBase(
      x, {}, false, static_cast<int>(ReduceType::kRedSum));
  // NOTE: reduce output is 0D tensor which has a dims_mapping as {}, while
  // output of squared_l2_norm is a tensor with shape [1] therefore it need to
  // have a dims_mapping as {-1}.
  auto& out_dist_dst = PADDLE_GET(TensorDistAttr, info.second[0]);
  out_dist_dst.set_dims_mapping({-1});
  return info;
}

}  // namespace distributed
}  // namespace phi
