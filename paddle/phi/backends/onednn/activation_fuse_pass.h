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

#pragma once
#include "paddle/fluid/framework/op_desc.h"

namespace phi {
namespace funcs {

static std::vector<std::string> GetSupportedActivations() {
  return std::vector<std::string>{"abs",
                                  "clip",
                                  "gelu",
                                  "hard_sigmoid",
                                  "hard_swish",
                                  "leaky_relu",
                                  "mish",
                                  "relu",
                                  "relu6",
                                  "sigmoid",
                                  "sqrt",
                                  "swish",
                                  "tanh"};
}

static std::unordered_map<std::string, std::string> GetAttributeMap(
    std::string act_type) {
  std::unordered_map<std::string, std::string> attr_map;
  if (act_type == "swish") {
    attr_map.emplace("beta", "fuse_alpha");
  } else if (act_type == "relu6") {
    attr_map.emplace("threshold", "fuse_alpha");
  } else if (act_type == "hard_sigmoid") {
    attr_map.emplace("slope", "fuse_alpha");
    attr_map.emplace("offset", "fuse_beta");
  } else if (act_type == "clip") {
    attr_map.emplace("min", "fuse_alpha");
    attr_map.emplace("max", "fuse_beta");
  } else {
    attr_map.emplace("alpha", "fuse_alpha");
    attr_map.emplace("beta", "fuse_beta");
  }
  return attr_map;
}

static void SetActivationAttrs(paddle::framework::OpDesc* fused_op,
                               paddle::framework::OpDesc* act_op,
                               const std::string& act_type) {
  if (fused_op->HasAttr("use_mkldnn")) {
    PADDLE_ENFORCE(PADDLE_GET_CONST(bool, fused_op->GetAttr("use_mkldnn")),
                   phi::errors::PreconditionNotMet(
                       "oneDNN activation fuses require use_mkldnn=True"));
  }
  fused_op->SetAttr("use_mkldnn", true);

  auto attr_map = phi::funcs::GetAttributeMap(act_type);
  for (const auto& attr : attr_map) {
    if (act_op->HasAttr(attr.first)) {
      fused_op->SetAttr(attr.second, act_op->GetAttr(attr.first));
    }
  }

  if (act_type == "gelu" && act_op->HasAttr("approximate")) {
    std::string gelu_act_type =
        PADDLE_GET_CONST(bool, act_op->GetAttr("approximate")) ? "gelu_tanh"
                                                               : "gelu_erf";
    fused_op->SetAttr("fuse_activation", gelu_act_type);
  } else {
    fused_op->SetAttr("fuse_activation", act_type);
  }
}

}  // namespace funcs
}  // namespace phi
