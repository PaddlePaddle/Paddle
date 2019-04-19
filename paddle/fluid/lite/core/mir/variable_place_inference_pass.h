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
  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override;

 private:
  // Mark the place of input arguments.
  void MarkInputPlace(SSAGraph* graph) {
    for (const auto& v : graph->inputs()) {
      // the feed op might in the inputs
      if (v->IsInstruct()) {
        LOG(INFO) << "found kernel in inputs " << v->AsInstruct().op_type;
        continue;
      }

      auto& arg = v->AsArgument();
      arg.place.target = argument_default_target_;
      // the other place description can't be determined yet, until their first
      // usage by some kernel.
    }
  }

  void InferenceArgumentPlace(SSAGraph* graph) {
    LOG(INFO) << "param-type-registry:\n" << ParamTypeRegistry::Global();
    for (auto& x : graph->InstructTopologicalOrder()) {
      auto& inst = x->AsInstruct();
      CHECK(inst.place.is_valid())
          << "kernel's place should be set when loaded";
      // deal with inputs
      for (auto& arg_name : inst.op_info->input_argnames()) {
        auto type =
            ParamTypeRegistry::Global().Retrieve<ParamTypeRegistry::IO::kInput>(
                inst.place, inst.op_type, arg_name);
        CHECK(type) << "no param-type found for " << inst.op_type << ":"
                    << arg_name << " " << inst.place.DebugString();
        auto arg_names = inst.op_info->input_argument().at(arg_name);
        // check if inputs's place is set, if not set, update them with the
        // kernel's declaration.

        for (auto& arg_name : arg_names) {
          auto* node = graph->RetrieveArgument(arg_name);
          CHECK(node) << "argument " << arg_name << " not exists in the graph";
          auto& arg_node = node->AsArgument();
          if (arg_node.place.is_valid()) continue;
          UpdatePlace(&arg_node.place, type->tensor_place);
        }
      }

      for (auto& arg_name : inst.op_info->output_argnames()) {
        auto type = ParamTypeRegistry::Global()
                        .Retrieve<ParamTypeRegistry::IO::kOutput>(
                            inst.place, inst.op_type, arg_name);
        CHECK(type) << "no param-type found for " << inst.op_type << ":"
                    << arg_name << " " << inst.place.DebugString();
        auto arg_names = inst.op_info->output_argument().at(arg_name);
        // check if outputs's place is set, if not set, update them with the
        // kernel's declaration.

        for (auto& arg_name : arg_names) {
          auto* node = graph->RetrieveArgument(arg_name);
          CHECK(node) << "argument " << arg_name << " not exists in the graph";
          auto& arg_node = node->AsArgument();
          if (arg_node.place.is_valid()) continue;
          UpdatePlace(&arg_node.place, type->tensor_place);
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
