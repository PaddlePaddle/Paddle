// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature DistributedFusedLambInitOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "distributed_fused_lamb_init",
      {"Param", "Grad"},
      {"beta1", "beta2", "apply_weight_decay", "alignment", "rank", "nranks"},
      {"FP32FusedParam",
       "FP32FusedGrad",
       "FP16FusedParam",
       "FP16FusedGrad",
       "Moment1",
       "Moment2",
       "Beta1Pow",
       "Beta2Pow",
       "FusedParamOffsets",
       "FP32ShardFusedParamOffsets",
       "FP16ShardFusedParamOffsets",
       "ParamInfo",
       "ParamOrder",
       "ParamOut",
       "MasterParamOut",
       "GradOut",
       "GlobalScale",
       "Step"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(distributed_fused_lamb_init,
                           phi::DistributedFusedLambInitOpArgumentMapping);
