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

#include "paddle/fluid/framework/ir/delete_fill_constant_op_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
void FillConstData(phi::DenseTensor* out_t, T value) {
  auto output_data = out_t->mutable_data<T>(platform::CPUPlace());
  for (int i = 0; i < out_t->numel(); i++) {
    output_data[i] = value;
  }
}

void DeleteFillConstantOpPass::ApplyImpl(ir::Graph* graph) const {
  bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
  // Not support
  if (with_dynamic_shape) {
    return;
  }
  FusePassBase::Init("delete_fill_constant_op_pass", graph);
  GraphPatternDetector detector;
  auto fill_constant_op =
      detector.mutable_pattern()
          ->NewNode("fill_constant")
          ->assert_is_op("fill_constant")
          ->assert_is_not_op_input("ValueTensor")
          ->assert_is_not_op_input("str_value")
          ->assert_is_not_op_input("ShapeTensor")
          ->assert_is_not_op_input("ShapeTensorList")
          ->assert_more([&](Node* node) {
            return node->Op()
                       ->GetAttrIfExists<std::vector<int64_t>>("shape")
                       .size() == 1;
          });
  auto fill_constant_out =
      detector.mutable_pattern()
          ->NewNode("fill_constant_out")
          ->assert_is_op_output("fill_constant")
          ->assert_more([](Node* x) { return x->outputs.size() == 1UL; });
  auto next_op = detector.mutable_pattern()
                     ->NewNode("next_op")
                     ->assert_is_not_op_type("conditional_block")
                     ->assert_is_not_op_type("while");
  // Create the topological connections for the above pattern nodes.
  fill_constant_op->LinksTo({fill_constant_out});
  next_op->LinksFrom({fill_constant_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    Node* fill_constant_op_node = subgraph.at(fill_constant_op);
    Node* fill_constant_out_node = subgraph.at(fill_constant_out);
    // Get fill_constant's attr
    auto fill_constant = fill_constant_op_node->Op();
    auto value = PADDLE_GET_CONST(float, fill_constant->GetAttr("value"));
    auto shape =
        PADDLE_GET_CONST(std::vector<int64_t>, fill_constant->GetAttr("shape"));
    auto* scope = param_scope();
    auto fill_constant_out_desc = fill_constant_out_node->Var();
    fill_constant_out_desc->SetShape(shape);
    fill_constant_out_desc->SetPersistable(true);
    auto* fill_constant_out_tensor = scope->Var(fill_constant_out_desc->Name())
                                         ->GetMutable<phi::DenseTensor>();
    auto dtype =
        framework::TransToPhiDataType(fill_constant_out_desc->GetDataType());
    fill_constant_out_tensor->Resize(phi::make_ddim(shape));
    switch (dtype) {
      case paddle::experimental::DataType::BOOL:
        FillConstData<bool>(fill_constant_out_tensor, static_cast<bool>(value));
        break;
      case paddle::experimental::DataType::INT32:
        FillConstData<int32_t>(fill_constant_out_tensor,
                               static_cast<int32_t>(value));
        break;
      case paddle::experimental::DataType::INT64:
        FillConstData<int64_t>(fill_constant_out_tensor,
                               static_cast<int64_t>(value));
        break;
      case paddle::experimental::DataType::FLOAT32:
        FillConstData<float>(fill_constant_out_tensor,
                             static_cast<float>(value));
        break;
      default:
        LOG(WARNING) << "Unsupported dtype for fill_constant op: " << dtype;
        return;
    }
    // Remove links in graph
    GraphSafeRemoveNodes(graph, {fill_constant_op_node});
  };

  detector(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_fill_constant_op_pass,
              paddle::framework::ir::DeleteFillConstantOpPass);
