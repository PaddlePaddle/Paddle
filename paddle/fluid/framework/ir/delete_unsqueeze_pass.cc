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
  PATTERN_DECL_NODE(common_op);
  // declare variable node's name
  PATTERN_DECL_NODE(common_op_in);
  PATTERN_DECL_NODE(common_op_out);
};

PDNode *DeleteUnsqueeze::operator()(PDNode *persis_x) {
  
  auto assert_ops = std::unordered_set<std::string>{"unsqueeze2"};
  persis_x->assert_is_ops_input(assert_ops);

  auto *op = pattern->NewNode(common_op_repr());
  op->assert_has_n_inputs(1);
  op->assert_is_ops(assert_ops);

  auto *op_out = pattern->NewNode(common_op_out_repr())
                  ->assert_is_ops_output(assert_ops, "Out");

  op->LinksFrom({persis_x});
  op_out->LinksFrom({op});
  return op_out;
}

}  // namespace patterns

DeleteUnsqueezePass::DeleteUnsqueezePass() {
}


void DeleteUnsqueezePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("delete_unsqueeze2", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;

  auto *persis_x_node = gpd.mutable_pattern()
                ->NewNode("persis_x_node")
                ->AsInput()
                ->assert_is_persistable_var();

  patterns::DeleteUnsqueeze fused_pattern(gpd.mutable_pattern(), "delete_unsqueeze2");
  fused_pattern(persis_x_node);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(persis_x_node) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    VLOG(4) << "handle DeleteUnsqueeze fuse";

    GET_IR_NODE_FROM_SUBGRAPH(op, common_op, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(op_out, common_op_out, fused_pattern);

    auto x_shape = subgraph.at(persis_x_node)->Var()->GetShape();
    std::cout << subgraph.at(persis_x_node)->Name() << std::endl;
   

    auto out_shape = op_out->Var()->GetShape();
    auto* scope = param_scope();
    auto unsqz_out_desc = op_out->Var();
    auto* persis_x_tensor = scope->FindVar(subgraph.at(persis_x_node)->Name())->GetMutable<LoDTensor>();
    auto* persis_x_ptr = persis_x_tensor->mutable_data<float>(platform::CPUPlace());
    
    framework::Scope *new_scope = new framework::Scope();
    new_scope->Var(subgraph.at(persis_x_node)->Name());
    auto* x_tensor = new_scope->FindVar(subgraph.at(persis_x_node)->Name())->GetMutable<LoDTensor>();
    x_tensor->Resize(persis_x_tensor->dims());
    std::cout << persis_x_tensor->dims()[0] << std::endl;
    std::cout << persis_x_tensor->dims()[1] << std::endl;
    std::cout << persis_x_tensor->dims()[2] << std::endl;
    std::cout << persis_x_tensor->dims()[3] << std::endl;
    auto* x_ptr = x_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < x_tensor->numel(); i++) {
      x_ptr[i] = persis_x_ptr[i];
    }

    auto* iter_node = op;
    std::vector<std::unique_ptr<OperatorBase>> ops;
    std::unordered_set<const paddle::framework::ir::Node*> remove_nodes{subgraph.at(persis_x_node)};



// for Op Nodes, it return the op's ith output, and make sure all others outputs muse be no outputs
// for Var Nodes, it return 0th output if outputs.size() == 1 and then return nullptr
auto PickOneOut = [&](Node* node) -> Node* {
  if(node->outputs.size() == 1) {
    return node->outputs[0];
  }
  else if(node->outputs.size() == 0) {
    return nullptr;
  }
  else
  {
    if(node->IsOp())
    {
      for(size_t i = 0; i < node->outputs.size(); i++)
      {
        if(node->outputs[i]->outputs.size() >= 1) return node->outputs[i];
      }
    }
    else
    {
      return nullptr;
    }
  }
  return nullptr;
};
    auto* last_node = iter_node;
    while(iter_node)
    {
      last_node = iter_node;
      if (iter_node->IsOp() && iter_node->inputs.size() == 1) {
        remove_nodes.emplace(iter_node);
        for (auto in_node : iter_node->inputs) {
          new_scope->Var(in_node->Var()->Name());
          new_scope->FindVar(in_node->Var()->Name())->GetMutable<LoDTensor>();
          remove_nodes.emplace(in_node);
        }
        for (auto out_node : iter_node->outputs) {
          new_scope->Var(out_node->Var()->Name());
          new_scope->FindVar(out_node->Var()->Name())->GetMutable<LoDTensor>();
        }
        ops.emplace_back(paddle::framework::OpRegistry::CreateOp(*iter_node->Op()));
        iter_node = PickOneOut(iter_node);
      }
      else if(iter_node->IsVar()) {
        std::cout << iter_node->Var()->Name() << std::endl;
        iter_node = PickOneOut(iter_node);
      } else {
        break;
      }
    }
    std::cout <<last_node->Var()->Name() << std::endl;

    for (size_t i = 0; i < ops.size(); i++) {
      ops[i]->Run(*new_scope, platform::CPUPlace());
    }

if(0)    unsqz_out_desc->SetShape(out_shape);
   // unsqz_out_desc->SetPersistable(true);
    //auto* unsqz_out_tensor = scope->Var(unsqz_out_desc->Name())->GetMutable<LoDTensor>();
    //auto dtype = framework::TransToPhiDataType(unsqz_out_desc->GetDataType());
    //unsqz_out_tensor->Resize(phi::make_ddim(out_shape));
    //unsqz_out_tensor->mutable_data<float>(platform::CPUPlace());

    // Remove links in graph
    GraphSafeRemoveNodes(graph, remove_nodes);
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_unsqueeze_pass,
              paddle::framework::ir::DeleteUnsqueezePass);
