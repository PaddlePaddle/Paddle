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

#ifdef PADDLE_WITH_NGRAPH
#include <algorithm>
#include <functional>

#include "paddle/fluid/framework/ngraph_bridge.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace framework {

std::map<std::string,
         std::function<void(const std::shared_ptr<OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphBridge::NG_NODE_MAP = {};

void NgraphBridge::build_graph(const std::shared_ptr<OperatorBase>& op) {
  auto& op_type = op->Type();
  NG_NODE_MAP[op_type](op, ngb_node_map);
}

}  // namespace framework
}  // namespace paddle
#endif
