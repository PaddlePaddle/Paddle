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
#include <unordered_set>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_ops.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ngraph_helper.h"

constexpr int64_t kNoPadding = -1;

namespace paddle {
namespace operators {

bool NgraphBridge::isRegister(const std::string& str) {
  return ops::NgraphSingleton::Lookup(str);
}

bool NgraphBridge::isSupported(
    const std::unique_ptr<framework::OperatorBase>& op) {
  static std::unordered_set<std::string> skip_op_list{
      "reshape", "reshape2", "lookup_table", "lookup_table_grad"};
  bool result = true;
  auto& op_type = op->Type();
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  if (!isRegister(op_type)) {
    if (skip_op_list.count(op_type)) {
      if (op_type == "lookup_table" || op_type == "lookup_table_grad") {
        if (op_attrs.Get<bool>("is_sparse")) {
          result = false;
        }
      } else if ((op_type == "reshape") || (op_type == "reshape2")) {
        if (op->Input("Shape") != paddle::framework::kEmptyVarName) {
          result = false;
        }
      } else {
        result = false;
      }
    }
  } else {
    result = false;
  }
  return result;
}

void NgraphBridge::BuildNgNode(
    const std::shared_ptr<framework::OperatorBase>& op) {
  auto& op_type = op->Type();
  ops::NgraphSingleton::BuildNode(ngb_node_map_, op, op_type);
}

}  // namespace operators
}  // namespace paddle
