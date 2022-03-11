// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/infer_shape_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace framework {
namespace ir {

void InferShapePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferShapePass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // Make batch_size fixed
  bool need_infer_shape = false;
  auto ipu_backend = platform::ipu::IpuBackend::GetInstance();
  auto micro_batch_size = ipu_backend->GetIpuStrategy()->micro_batch_size;
  auto feed_list = Get<std::vector<std::string>>("feed_list");
  for (auto node : graph->Nodes()) {
    if (!node->IsVar()) {
      continue;
    }
    bool is_feed = std::find(feed_list.begin(), feed_list.end(),
                             node->Name()) != feed_list.end();
    if (is_feed) {
      auto input_shape = node->Var()->GetShape();
      if (input_shape[0] <= -1) {
        input_shape[0] = micro_batch_size;
        node->Var()->SetShape(input_shape);
        need_infer_shape = true;
      }
      // int64->int32
      if (node->Var()->GetDataType() == proto::VarType::INT64) {
        node->Var()->SetDataType(proto::VarType::INT32);
      }
    }
  }

  // temp scope for shape inference
  if (need_infer_shape) {
    std::shared_ptr<paddle::framework::Scope> scope(
        new paddle::framework::Scope());
    for (auto node : graph->Nodes()) {
      if (!node->IsVar()) {
        continue;
      }
      auto var_desc = node->Var();
      auto* ptr = scope->Var(var_desc->Name());
      paddle::framework::InitializeVariable(ptr, var_desc->GetType());

      auto tensor = ptr->GetMutable<paddle::framework::LoDTensor>();
      tensor->Resize(phi::make_ddim(var_desc->GetShape()));
    }

    // infer shape
    auto nodes = ir::TopologySortOperations(*graph);
    for (auto node : nodes) {
      VLOG(10) << "InferShapePass: Infer shape for Op (" << node->Name() << ")";
      auto op_desc = node->Op();
      if (op_desc->Type() == "popart_optimizer") {
        continue;
      }
      auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
      paddle::framework::RuntimeContext ctx(op->Inputs(), op->Outputs(),
                                            *scope);
      op->RuntimeInferShape(*scope, paddle::platform::CPUPlace(), ctx);

      for (auto it = ctx.outputs.begin(); it != ctx.outputs.end(); it++) {
        for (int i = 0; i < it->second.size(); i++) {
          auto output_name = op_desc->Output(it->first)[i];
          auto dim =
              it->second[i]->GetMutable<paddle::framework::LoDTensor>()->dims();
          auto new_shape = phi::vectorize(dim);
          for (auto output_node : node->outputs) {
            if (output_node->Name() == output_name) {
              output_node->Var()->SetShape(new_shape);
              if (VLOG_IS_ON(10)) {
                std::ostringstream sout;
                sout << "InferShapePass: output[" << output_node->Name()
                     << "], infer shape:[";
                for (auto s : new_shape) {
                  sout << std::to_string(s) << ", ";
                }
                sout << "]";
                VLOG(10) << sout.str();
              }
            }
          }
        }
      }
      VLOG(10) << "InferShapePass: Infer shape for Op (" << node->Name()
               << ") finished";
    }
    // release the temp scope
    scope.reset();
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave InferShapePass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(infer_shape_pass, paddle::framework::ir::InferShapePass)
    .RequirePassAttr("feed_list");
