/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/placement_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Specifies which operators should use MKLDNN.
 */
class MKLDNNPlacementPass : public PlacementPassBase {
 protected:
  bool IsSupport(const Node* op) const override;

 private:
  const std::string GetPlacementName() const override { return "MKLDNN"; }

  const std::string GetAttrName() const override { return "use_mkldnn"; }

  const std::unordered_set<std::string> GetOpTypesList() const override {
    return Get<std::unordered_set<std::string>>("mkldnn_enabled_op_types");
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
