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

KernelSignature OneHotOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("depth_tensor")) {
    return KernelSignature("one_hot_raw",
                           {"X"},
                           {"depth_tensor", "dtype", "allow_out_of_range"},
                           {"Out"});
  } else {
    return KernelSignature("one_hot_raw",
                           {"X"},
                           {"depth", "dtype", "allow_out_of_range"},
                           {"Out"});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(one_hot_v2, one_hot);

PD_REGISTER_ARG_MAPPING_FN(one_hot_v2, phi::OneHotOpArgumentMapping);
