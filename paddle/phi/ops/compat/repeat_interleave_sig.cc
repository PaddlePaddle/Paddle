// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature RepeatInterleaveOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("RepeatsTensor")) {
    VLOG(3) << "sig------ repeat_interleave_with_tensor_index";
    return KernelSignature("repeat_interleave_with_tensor_index",
                           {"X", "RepeatsTensor"},
                           {"dim"},
                           {"Out"});
  } else {
    VLOG(3) << "sig ------repeat_interleave";
    return KernelSignature(
        "repeat_interleave", {"X"}, {"Repeats", "dim"}, {"Out"});
  }
}

KernelSignature RepeatInterleaveGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("RepeatsTensor")) {
    VLOG(3) << "sig ------repeat_interleave with tensor grad";
    return KernelSignature("repeat_interleave_with_tensor_index_grad",
                           {"X", "RepeatsTensor", "Out@GRAD"},
                           {"dim"},
                           {"X@GRAD"});
  } else {
    VLOG(3) << "sig repeat_interleave grad";
    return KernelSignature("repeat_interleave_grad",
                           {"X", "Out@GRAD"},
                           {"Repeats", "dim"},
                           {"X@GRAD"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(repeat_interleave,
                           phi::RepeatInterleaveOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(repeat_interleave_grad,
                           phi::RepeatInterleaveGradOpArgumentMapping);
