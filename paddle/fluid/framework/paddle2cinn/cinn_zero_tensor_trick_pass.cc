/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/paddle2cinn/cinn_zero_tensor_trick_pass.h"

#include <string>
#include "glog/logging.h"

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

void CinnZeroTensorTrickPass::ApplyImpl(ir::Graph* graph) const {
  // fix shape attr of these ops
  const std::unordered_set<std::string> op_cases_fix_attr{"fill_constant",
                                                          "uniform_random",
                                                          "expand_v2",
                                                          "assign_value",
                                                          "gaussian_random",
                                                          "set_value"};
  for (const ir::Node* n : graph->Nodes()) {
    if (n->IsOp() && op_cases_fix_attr.count(n->Op()->Type())) {
      if (n->Op()->HasAttr("shape")) {
        auto attr_type = n->Op()->GetAttrType("shape");
        if (attr_type == paddle::framework::proto::INTS) {
          auto shapes =
              PADDLE_GET_CONST(std::vector<int32_t>, n->Op()->GetAttr("shape"));
          if (shapes.empty()) {
            shapes.push_back(1);
            n->Op()->SetAttr("shape", shapes);
            VLOG(4) << "op " << n->Op()->Type()
                    << " shape attribute dims is empty, fix dim -> {1} ";
          }
        } else { /* attr_type == paddle::framework::proto::LONGS */
          auto shapes =
              PADDLE_GET_CONST(std::vector<int64_t>, n->Op()->GetAttr("shape"));
          if (shapes.empty()) {
            shapes.push_back(1);
            n->Op()->SetAttr("shape", shapes);
            VLOG(4) << "op " << n->Op()->Type()
                    << " shape attribute dims is empty, fix dim -> {1} ";
          }
        }
      }
    }
    if (n->IsVar()) {
      if (n->Var() && n->Var()->GetType() == proto::VarType::LOD_TENSOR) {
        std::vector<int64_t> shape = n->Var()->GetShape();
        if (shape.empty()) {
          shape.push_back(1);
          n->Var()->SetShape(shape);
          VLOG(4) << "var " << n->Name() << " dims is empty, fix dim -> {1} ";
        }
      }
    }
  }
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cinn_zero_tensor_trick_pass,
              paddle::framework::paddle2cinn::CinnZeroTensorTrickPass);
