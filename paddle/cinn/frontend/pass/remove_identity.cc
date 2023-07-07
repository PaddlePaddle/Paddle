// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

#define SHAPE_SAME_REMOVE(op_name)                 \
  {                                                \
#op_name, [](const Instruction& instr) -> bool { \
    const auto& input_shape = instr->inputs[0]->shape; \
    const auto& output_shape = instr->outputs[0]->shape; \
    return input_shape == output_shape; \
  } \
  }

static std::unordered_map<std::string, std::function<bool(const Instruction&)>>
    identity_ops = {
        {"identity", [](const Instruction& instr) -> bool { return true; }},
        {"scale",
         [](const Instruction& instr) -> bool {
           bool bias_zero = !instr->attrs.count("bias") ||
                            instr.GetAttrs<float>("bias") == 0.0f;
           bool scale_one = !instr->attrs.count("scale") ||
                            instr.GetAttrs<float>("scale") == 1.0f;
           return bias_zero && scale_one;
         }},
        {"cast",
         [](const Instruction& instr) -> bool {
           const auto& input_dtype = instr->inputs[0]->type;
           const auto& output_dtype = instr->outputs[0]->type;
           return input_dtype == output_dtype;
         }},
        {"transpose",
         [](const Instruction& instr) -> bool {
           const auto& input_shape = instr->inputs[0]->shape;
           const auto& axis = instr.GetAttrs<std::vector<int>>("axis");

           bool can_remove = (input_shape.size() == axis.size());
           if (can_remove) {
             for (int i = 0; i < axis.size(); ++i) {
               if (axis[i] != i) {
                 can_remove = false;
                 break;
               }
             }
           }

           return can_remove;
         }},
        {"concat",
         [](const Instruction& instr) -> bool {
           return (instr->inputs.size() == 1);
         }},
        {"split",
         [](const Instruction& instr) -> bool {
           return (instr->outputs.size() == 1);
         }},
        SHAPE_SAME_REMOVE(broadcast_to),
        SHAPE_SAME_REMOVE(reduce_sum),
        SHAPE_SAME_REMOVE(reduce_prod),
        SHAPE_SAME_REMOVE(reduce_max),
        SHAPE_SAME_REMOVE(reduce_min),
        SHAPE_SAME_REMOVE(reduce_all),
        SHAPE_SAME_REMOVE(reduce_any),
        SHAPE_SAME_REMOVE(slice),
        SHAPE_SAME_REMOVE(reshape)};

#undef SHAPE_SAME_REMOVE

namespace {
bool check_reduce_to_reshape(const Instruction& instr) {
  const auto& input_shape = instr->inputs[0]->shape;

  auto dims = instr->attrs.count("dim")
                  ? instr.GetAttrs<std::vector<int>>("dim")
                  : std::vector<int>();

  if (dims.empty()) {
    for (int i = 0; i < input_shape.size(); ++i) {
      dims.emplace_back(i);
    }
  }

  for (auto aixs : dims) {
    if (input_shape[aixs] != 1) {
      return false;
    }
  }
  return true;
}
}  // namespace

static std::unordered_map<std::string, std::function<bool(const Instruction&)>>
    reshape_ops = {{"reduce_sum", check_reduce_to_reshape},
                   {"reduce_prod", check_reduce_to_reshape},
                   {"reduce_max", check_reduce_to_reshape},
                   {"reduce_min", check_reduce_to_reshape},
                   {"reduce_all", check_reduce_to_reshape},
                   {"reduce_any", check_reduce_to_reshape}};

// RemoveIdentityPass will remove the identity instructions in following
// patterns:
//
// 1. When varB is not in fetch_ids, the identity and varB will be removed.
//    When varB is in fetch_ids and varA is not in fetch_ids, the identity and
//    varA will be removed.
//        instrA                      instrA
//          | varA                      |
//      identity           =>           | varA/varB
//          | varB                      |
//        instrB                      instrB
//
// 2. Multiply outputs are also supported.
//        instrA                      instrA
//          | varA                      |
//      identity           =>           | varA/varB
//          | varB                      |
//         / \                         / \
//   instrB   instrC             instrB   instrC
class RemoveIdentityPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    CollectInfo(*program, fetch_ids);

    VLOG(3) << "Total remove " << remove_idxs_.size() << " instructions.";

    NetBuilder builder("remove_identity_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); ++i) {
      if (remove_idxs_.count(i)) {
        continue;
      }

      auto& instr = (*program)[i];
      if (replace_identity_idxs_.count(i)) {
        VLOG(4) << "Replace op " << instr->outputs[0]->id << "["
                << cinn::utils::Join(instr->outputs[0]->shape, ", ")
                << "]=" << instr->op_type << "{" << instr->inputs[0]->id << "["
                << cinn::utils::Join(instr->inputs[0]->shape, ", ")
                << "]} to identity";

        instr->op_type = "identity";
        instr->attrs.clear();
      } else if (reshape_ops.count(instr->op_type) &&
                 reshape_ops.at(instr->op_type)(instr)) {
        VLOG(4) << "Replace op " << instr->outputs[0]->id << "["
                << cinn::utils::Join(instr->outputs[0]->shape, ", ")
                << "]=" << instr->op_type << "{" << instr->inputs[0]->id << "["
                << cinn::utils::Join(instr->inputs[0]->shape, ", ")
                << "]} to reshape";

        instr->op_type = "reshape";
        instr->attrs.clear();
        instr->attrs["shape"] = instr->outputs[0]->shape;
      }

      auto& inputs = instr->inputs;
      for (size_t j = 0; j < inputs.size(); ++j) {
        if (origin2new_.count(inputs[j].get())) {
          inputs[j] = origin2new_.at(inputs[j].get());
        }
      }
      auto& outputs = instr->outputs;
      for (size_t j = 0; j < outputs.size(); ++j) {
        if (origin2new_.count(outputs[j].get())) {
          outputs[j] = origin2new_.at(outputs[j].get());
        }
      }
      builder.AppendInstruction(instr);
    }
    *program = builder.Build();
  }

  void Clear() override {
    remove_idxs_.clear();
    origin2new_.clear();
    replace_identity_idxs_.clear();
  }

 private:
  void CollectInfo(const Program& program,
                   const std::unordered_set<std::string>& fetch_ids) {
    remove_idxs_.clear();
    origin2new_.clear();

    std::unordered_set<std::string> feed_ids;
    for (auto& var : program.GetInputs()) {
      feed_ids.insert(var->id);
    }
    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      if (!identity_ops.count(instr->op_type)) {
        continue;
      }

      if (!identity_ops.at(instr->op_type)(instr)) {
        continue;
      }
      CHECK_EQ(instr->inputs.size(), 1)
          << instr->op_type << " should have only 1 input. But here " << instr;
      CHECK_EQ(instr->outputs.size(), 1)
          << instr->op_type << " should have only 1 output. But here " << instr;

      auto& input_var = instr->inputs[0];
      auto& output_var = instr->outputs[0];

      bool can_input_var_removed =
          !feed_ids.count(input_var->id) && !fetch_ids.count(input_var->id);
      bool can_output_var_removed = !fetch_ids.count(output_var->id);
      if (can_input_var_removed || can_output_var_removed) {
        bool updated = false;
        if (can_output_var_removed) {
          updated = UpdateOrigin2New(output_var, input_var);
        }
        if (!updated && can_input_var_removed) {
          updated = UpdateOrigin2New(input_var, output_var);
        }
        if (updated) {
          VLOG(3) << "Remove the " << i << "-th instruction: " << instr;
          remove_idxs_.insert(i);
        }
      } else {
        replace_identity_idxs_.insert(i);
      }
    }

    for (auto& v : origin2new_) {
      const auto& reserved_var = v.second;
      auto iter = origin2new_.find(reserved_var.get());
      if (iter != origin2new_.end()) {
        VLOG(4) << "Update " << v.first->id << " -> " << reserved_var->id
                << " to " << v.first->id << " -> " << iter->second->id;
        origin2new_[v.first] = iter->second;
      }
    }

    VLOG(4) << "origin2new_ {";
    for (auto& iter : origin2new_) {
      VLOG(4) << "  " << iter.first->id << " -> " << iter.second->id;
    }
    VLOG(4) << "}";
  }

  bool UpdateOrigin2New(const Variable& origin, const Variable& new_var) {
    if (!origin2new_.count(origin.get())) {
      if (origin2new_.count(new_var.get())) {
        VLOG(4) << "Add " << origin->id << " -> "
                << origin2new_[new_var.get()]->id;
        origin2new_.emplace(origin.get(), origin2new_[new_var.get()]);
      } else {
        VLOG(4) << "Add " << origin->id << " -> " << new_var->id;
        origin2new_.emplace(origin.get(), new_var);
      }
      return true;
    }
    return false;
  }

  std::unordered_set<int> remove_idxs_;
  std::unordered_map<_Variable_*, Variable> origin2new_;
  std::unordered_set<int> replace_identity_idxs_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(RemoveIdentity) {
  CINN_REGISTER_PROGRAM_PASS(RemoveIdentity,
                             cinn::frontend::pass::RemoveIdentityPass);

  return true;
}
