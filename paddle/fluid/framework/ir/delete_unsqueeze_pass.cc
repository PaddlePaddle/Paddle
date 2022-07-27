/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/delete_unsqueeze_pass.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include <vector>

#include "paddle/fluid/framework/convert_utils.h"

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

struct DeleteUnsqueeze : public PatternBase {
  DeleteUnsqueeze(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "delete_unsqueeze_pass") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(unsqz);
  // declare variable node's name
  PATTERN_DECL_NODE(unsqz_in);
  PATTERN_DECL_NODE(unsqz_out);
};

PDNode *DeleteUnsqueeze::operator()(PDNode *x) {
  
  auto assert_ops = std::unordered_set<std::string>{"unsqueeze2", "reshape2"};
  x->assert_is_ops_input(assert_ops);

  auto *unsqz = pattern->NewNode(unsqz_repr());
  unsqz->assert_has_n_inputs(1);
  unsqz->assert_is_ops(assert_ops);

  auto *unsqz_out = pattern->NewNode(unsqz_out_repr())
                  ->assert_is_ops_output(assert_ops, "Out");

  unsqz->LinksFrom({x});
  unsqz_out->LinksFrom({unsqz});
  return unsqz_out;
}

}  // namespace patterns

DeleteUnsqueezePass::DeleteUnsqueezePass() {
}

static Node* PickOneOut(std::vector<Node*> outputs)
{
  if (outputs.size() == 1)
  {
    return outputs[0];
  }
  else
  {
    for (auto node : outputs)
    {
      if(node->outputs.size())
      {
        return node;
      }
    }
  }
  return nullptr;
}

void DeleteUnsqueezePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("delete_unsqueeze2", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;

  auto *unsqueeze2_x = gpd.mutable_pattern()
                ->NewNode("unsqueeze2_x")
                ->AsInput()
                ->assert_is_persistable_var();

  patterns::DeleteUnsqueeze fused_pattern(gpd.mutable_pattern(), "delete_unsqueeze2");
  fused_pattern(unsqueeze2_x);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(unsqueeze2_x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    VLOG(4) << "handle DeleteUnsqueeze fuse";

    GET_IR_NODE_FROM_SUBGRAPH(unsqz_op, unsqz, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqz_out, unsqz_out, fused_pattern);

    auto x_shape = subgraph.at(unsqueeze2_x)->Var()->GetShape();
    std::cout << subgraph.at(unsqueeze2_x)->Name() << std::endl;

    auto out_shape = unsqz_out->Var()->GetShape();
    auto* scope = param_scope();
     auto unsqz_out_desc = unsqz_out->Var();
    framework::Scope *new_scope = new framework::Scope();
    new_scope->Var(subgraph.at(unsqueeze2_x)->Name());
    for (auto out_node : unsqz_op->outputs) {
      new_scope->Var(out_node->Var()->Name());
      new_scope->FindVar(out_node->Var()->Name())->GetMutable<LoDTensor>();
    }

    auto* persis_x_tensor = scope->FindVar(subgraph.at(unsqueeze2_x)->Name())->GetMutable<LoDTensor>();
    auto* persis_x_ptr = persis_x_tensor->mutable_data<float>(platform::CPUPlace());

    auto* x_tensor = new_scope->FindVar(subgraph.at(unsqueeze2_x)->Name())->GetMutable<LoDTensor>();
    x_tensor->Resize(phi::make_ddim(std::vector<int64_t>{16,144,144}));
    auto* x_ptr = x_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < x_tensor->numel(); i++) {
      x_ptr[i] = persis_x_ptr[i];
    }

    std::unique_ptr<OperatorBase> run_op = paddle::framework::OpRegistry::CreateOp(*unsqz_op->Op());
    run_op->Run(*new_scope, platform::CPUPlace());
    auto* y_tensor = new_scope->FindVar(unsqz_out_desc->Name())->GetMutable<LoDTensor>();

    std::cout << y_tensor->dims()[0] << std::endl;
    std::cout << y_tensor->dims()[1] << std::endl;
    std::cout << y_tensor->dims()[2] << std::endl;
    std::cout << y_tensor->dims()[3] << std::endl;

    auto* iter_node = PickOneOut(unsqz_out->outputs);
    while(false && iter_node)
    {
      if (iter_node->IsOp() && iter_node->inputs.size() == 1)
      {
        std::cout << iter_node->Op()->Type() << std::endl;
        std::unique_ptr<OperatorBase> run_op = paddle::framework::OpRegistry::CreateOp(*iter_node->Op());
        run_op->Run(*scope, platform::CPUPlace());
        iter_node = PickOneOut(iter_node->outputs);
      }
      else if(iter_node->IsVar()) {
        std::cout << iter_node->Var()->Name() << std::endl;
        iter_node = PickOneOut(iter_node->outputs);
      } else {
        break;
      }
    }

if(0)    unsqz_out_desc->SetShape(out_shape);
   // unsqz_out_desc->SetPersistable(true);
    //auto* unsqz_out_tensor = scope->Var(unsqz_out_desc->Name())->GetMutable<LoDTensor>();
    //auto dtype = framework::TransToPhiDataType(unsqz_out_desc->GetDataType());
    //unsqz_out_tensor->Resize(phi::make_ddim(out_shape));
    //unsqz_out_tensor->mutable_data<float>(platform::CPUPlace());

    // Remove links in graph
    //GraphSafeRemoveNodes(graph, {unsqz_op});
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_unsqueeze_pass,
              paddle::framework::ir::DeleteUnsqueezePass);
