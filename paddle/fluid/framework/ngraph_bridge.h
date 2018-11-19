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

#ifdef PADDLE_WITH_NGRAPH

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace framework {

class NgraphBridge {
 public:
  static std::map<
      std::string,
      std::function<void(const std::shared_ptr<OperatorBase>&,
                         std::shared_ptr<std::unordered_map<
                             std::string, std::shared_ptr<ngraph::Node>>>)>>
      NG_NODE_MAP;

  explicit NgraphBridge(
      std::shared_ptr<
          std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
          var_node_map)
      : ngb_node_map(var_node_map) {}

  void build_graph(const std::shared_ptr<OperatorBase>& op);

 private:
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      ngb_node_map;
};

}  // namespace framework
}  // namespace paddle
#endif
