// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Mark the place of the variables in the SSAGrpah, it will inference the
 * variables' place by the kernels outputs them.
 */
class VariablePlaceInferencePass : public DebugPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  // Mark the place of input arguments.
  void MarkInputPlace(SSAGraph* graph) {
    CHECK(!graph->inputs().empty()) << "graph's inputs should be set";
    for (const auto& v : graph->inputs()) {
      // the feed op might in the inputs
      if (v->IsStmt()) {
        LOG(INFO) << "found kernel in inputs " << v->AsStmt().op_type();
        continue;
      }
    }
  }

  void CheckAllArgumentTypeDetermined(SSAGraph* graph) {
    for (auto& node : graph->mutable_nodes()) {
      if (node.IsArg()) {
        CHECK(node.AsArg().type) << "node " << node.AsArg().name
                                 << " type not determined, " << &node;
      }
    }
  }

  void InferenceArgumentPlace(SSAGraph* graph) {
    VLOG(3) << "param-type-registry:\n" << ParamTypeRegistry::Global();
    for (auto& x : graph->StmtTopologicalOrder()) {
      auto& inst = x->AsStmt();
      // The IoCopyOp is a tool operator, it won't support the type inference.
      if (inst.op_type() == "io_copy") continue;
      // LOG(INFO) << "- inferencing type " <<
      // deal with inputs
      VLOG(4) << "Infering op " << inst.op_info()->Repr();
      // TODO(zhaolong): Add check if the node's name in op's arguments.

      auto get_argname = [&](
          const std::string& node_name,
          const std::map<std::string, std::vector<std::string>>& argname_map)
          -> std::string {
            for (auto& ele : argname_map) {
              auto it =
                  std::find(ele.second.begin(), ele.second.end(), node_name);
              if (it != ele.second.end()) return ele.first;
            }
            return "";
          };

      for (auto* x_in : x->inlinks) {
        std::string node_name = x_in->AsArg().name;
        std::string arg_name = get_argname(node_name, inst.op_info()->inputs());
        CHECK(arg_name.size() > 0) << "can not found op arguments for node "
                                   << node_name;
        VLOG(3) << "-- input arg_name " << arg_name;
        auto type = inst.picked_kernel().GetInputDeclType(arg_name);
        if (!x_in->AsArg().type) {
          VLOG(4) << "set type " << *type << " " << x_in;
          x_in->AsArg().type = type;
        }
      }

      VLOG(3) << "inst " << inst.op_info()->Repr();
      for (auto* x_out : x->outlinks) {
        std::string node_name = x_out->AsArg().name;
        std::string arg_name =
            get_argname(node_name, inst.op_info()->outputs());
        CHECK(arg_name.size() > 0) << "can not found op arguments for node "
                                   << node_name << " in Inst "
                                   << inst.op_type();
        VLOG(3) << "-- output arg_name " << arg_name;
        auto type = inst.picked_kernel().GetOutputDeclType(arg_name);
        if (!x_out->AsArg().type) {
          VLOG(4) << "set type " << *type << " " << x_out;
          x_out->AsArg().type = type;
        }
      }
    }
  }

  // Update me's kUnk fields by other's fields.
  void UpdatePlace(Place* me, const Place& other) {
    CHECK(other.is_valid());
    if (me->target == TARGET(kUnk)) {
      me->target = other.target;
    }
    if (me->precision == PRECISION(kUnk)) {
      me->precision = other.precision;
    }
    if (me->layout == DATALAYOUT(kUnk)) {
      me->layout = other.layout;
    }
  }

 private:
  // The default target for arguments, e.g. load weights to CPU memory for CUDA
  // computation by default.
  TargetType argument_default_target_{TARGET(kHost)};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
