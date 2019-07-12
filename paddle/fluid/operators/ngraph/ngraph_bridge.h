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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "ngraph/node.hpp"

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class NgraphBridge {
 public:
  explicit NgraphBridge(
      std::shared_ptr<
          std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
          var_node_map)
      : ngb_node_map_(var_node_map) {}

  void BuildNgNode(const std::shared_ptr<framework::OperatorBase>& op);

  static bool isRegister(const std::string& str);

  static bool isSupported(const std::unique_ptr<framework::OperatorBase>& op);

 private:
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      ngb_node_map_;
};

}  // namespace operators
}  // namespace paddle
