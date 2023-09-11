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

KernelSignature PRecvOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{};
  paddle::small_vector<const char*> attrs;

  attrs.emplace_back("peer");
  attrs.emplace_back("dtype");
  paddle::small_vector<const char*> outputs{"out"};

  if (ctx.IsDenseTensorOutput("X")) {
    attrs.emplace_back("dynamic_shape");
    return KernelSignature(
        "p_recv", std::move(inputs), std::move(attrs), std::move(outputs));
  } else if (ctx.IsDenseTensorVectorOutput("X")) {
    attrs.emplace_back("out_shape");
    return KernelSignature("p_recv_array",
                           std::move(inputs),
                           std::move(attrs),
                           std::move(outputs));
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(p_recv, phi::PRecvOpArgumentMapping);
