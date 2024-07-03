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
  VLOG(4) << "Using ReductionInferSpmd Rule as interal implement.";
  return ReductionInferSpmdBase(
      x, {}, false, static_cast<int>(ReduceType::kRedSum));
}

}  // namespace distributed
}  // namespace phi
