// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature SendV2OpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("send_v2_phi", {"X"}, {"peer", "dynamic_shape"}, {});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(send_v2, send_v2_phi);
PD_REGISTER_ARG_MAPPING_FN(send_v2, phi::SendV2OpArgumentMapping);
