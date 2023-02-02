/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/record_reshape_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/inference/utils/io_utils.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct RecordReshapePass : public PatternBase {
  RecordReshapePass(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "record_reshape_pass") {}
};
}  // namespace patterns

RecordReshapePass::RecordReshapePass() {}

void RecordReshapePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("constant_folding", graph);
  auto *scope = param_scope();

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "scope must not be null when applying constant floding."));

  std::map<std::string, std::vector<int>> min_input_shape;
  std::map<std::string, std::vector<int>> max_input_shape;
  std::map<std::string, std::vector<int>> opt_input_shape;
  std::map<std::string, std::vector<int>> min_shape_tensor;
  std::map<std::string, std::vector<int>> max_shape_tensor;
  std::map<std::string, std::vector<int>> opt_shape_tensor;
  auto shape_range_info_path_ = Get<std::string>("trt_shape_range_info_path");
  ;
  if (shape_range_info_path_.size()) {
    inference::DeserializeShapeRangeInfo(shape_range_info_path_,
                                         &min_input_shape,
                                         &max_input_shape,
                                         &opt_input_shape,
                                         &min_shape_tensor,
                                         &max_shape_tensor,
                                         &opt_shape_tensor);
  }

  auto op_node_sorted = framework::ir::TopologyVarientSort(
      *graph, static_cast<framework::ir::SortKind>(0));
  for (auto *op_node : op_node_sorted) {
    if (!op_node->IsOp()) continue;
    if (op_node->Name() != "reshape2") {
      continue;
    }

    std::string out_var_name = op_node->Op()->Output("Out").front();

    int rank = min_input_shape[out_var_name].size();
    std::vector<int> shape;
    for (int i = 0; i < rank; i++) {
      if (min_input_shape[out_var_name][i] ==
          max_input_shape[out_var_name][i]) {
        shape.push_back(min_input_shape[out_var_name][i]);
      } else {
        shape.push_back(-1);
      }
    }
    op_node->Op()->SetAttr("compiled_time_shape", shape);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(record_reshape_pass, paddle::framework::ir::RecordReshapePass);
