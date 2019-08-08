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

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_ops.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {

bool NgraphBridge::isRegister(const std::string& str) {
  return ops::NgraphSingleton::Lookup(str);
}

void NgraphBridge::BuildNgNode(
    const std::shared_ptr<framework::OperatorBase>& op) {
  auto& op_type = op->Type();
  ops::NgraphSingleton::BuildNode(ngb_node_map_, op, op_type);
}

}  // namespace operators
}  // namespace paddle
