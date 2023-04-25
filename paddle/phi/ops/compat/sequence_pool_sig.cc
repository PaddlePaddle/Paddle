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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature SequencePoolOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("sequence_pool",
                         {"X"},
                         {"is_test", "pooltype", "pad_value"},
                         {"Out", "MaxIndex"});
}

KernelSignature SequencePoolGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& use_pooltype_maxindex =
      paddle::any_cast<std::string>(ctx.Attr("pooltype"));

  if (use_pooltype_maxindex == "MAX") {
    return KernelSignature("sequence_pool_grad",
                           {"X", "MaxIndex", "Out@GRAD"},
                           {"is_test", "pooltype", "pad_value"},
                           {"X@GRAD"});
  } else {
    return KernelSignature("sequence_pool_grad",
                           {"X", "Out@GRAD"},
                           {"is_test", "pooltype", "pad_value"},
                           {"X@GRAD"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(sequence_pool, phi::SequencePoolOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sequence_pool_grad,
                           phi::SequencePoolGradOpArgumentMapping);
