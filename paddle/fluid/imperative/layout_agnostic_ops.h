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

#include <unordered_set>
#include "paddle/fluid/framework/op_info.h"

namespace paddle {
namespace imperative {

class LayoutAutotuneOperators {
 public:
  static LayoutAutotuneOperators& Instance() {
    static LayoutAutotuneOperators layout_autoTune_ops;
    return layout_autoTune_ops;
  }

  std::unordered_set<std::string> GetAgnosticOps() { return agnostic_ops_; }

 private:
  LayoutAutotuneOperators() {
    const auto& op_info = paddle::framework::OpInfoMap::Instance().map();
    for (auto it = op_info.begin(); it != op_info.end(); it++) {
      // only record forwrd operators
      if (it->first.find("_grad") != std::string::npos) {
        continue;
      }

      // some normalization operators such as instance_norm and layer_norm
      // do not have data_format attr, but are layout sensitive.
      if (it->first.find("norm") != std::string::npos) {
        continue;
      }

      // The elementwise ops is agnostic ops, but has axis attr
      if (it->first.find("elementwise") != std::string::npos ||
          it->first.find("less_") != std::string::npos ||
          it->first.find("greater_") != std::string::npos ||
          it->first.find("equal") != std::string::npos) {
        agnostic_ops_.emplace(it->first);
      }

      bool layout_sensitive = false;
      auto* attr_checker = it->second.Checker();
      if (attr_checker) {
        auto attrs = attr_checker->GetDefaultAttrMap();
        // Attribute name is fuzzy matched
        for (auto& attr : attrs) {
          auto attr_name = attr.first;
          VLOG(6) << "OP: " << it->first << " Attr Name: " << attr_name;
          if (attr_name.find("data_format") != std::string::npos ||
              attr_name.find("data_layout") != std::string::npos ||
              attr_name.find("axis") != std::string::npos ||
              attr_name.find("axes") != std::string::npos ||
              attr_name.find("dim") != std::string::npos ||
              attr_name.find("start") != std::string::npos ||
              attr_name.find("end") != std::string::npos) {
            layout_sensitive = true;
            break;
          }
        }
      }

      if (!layout_sensitive) {
        VLOG(4) << "agnostic_ops: " << it->first;
        agnostic_ops_.emplace(it->first);
      }
    }
  }

  std::unordered_set<std::string> agnostic_ops_;
};

}  // namespace imperative
}  // namespace paddle
