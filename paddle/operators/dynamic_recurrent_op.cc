/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve .

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/dynamic_recurrent_op.h"

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class DynamicRecurrentAlgorithmProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  DynamicRecurrentAlgorithmProtoAndCheckerMaker(
      framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name = DynamicRecurrentOp::kArgName;
    // inputs and outputs stored in proto
    AddInput(name.inlinks,
             "the inputs that need to be segmented for each step.")
        .AsDuplicable();
    AddInput(name.boot_memories, "variables to initialize memories.")
        .AsDuplicable();

    AddOutput(name.outlinks, "the outputs that need to concated for all steps.")
        .AsDuplicable();
    AddOutput(name.step_scopes, "step scopes");

    // Attributes stored in AttributeMap
    AddAttr<std::vector<std::string>>(name.pre_memories,
                                      "names of pre-memories");
    AddAttr<std::vector<std::string>>(name.memories, "names of memories");

    AddComment("This is a recurrent group operator.");
  }
};

void DynamicRecurrentOp::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  cache_.Init(kArgName, *this, scope, &arg_);
  SplitInputs();
  CreateScopes();
  WriteStepInputs();
  InitStates();

  for (size_t step = 0; step < cache_.num_steps; step++) {
    // call stepnet
    stepnet_->Run(scope, dev_ctx);
  }

  WriteStepOutputs();
  ConcatOutputs();
}

void DynamicRecurrentOp::SplitInputs() const {
  // TODO(superjom) make level a config
  // TODO(superjom) check all the inputs has the same LoD
  int level = 0;
  const auto& inlinks = cache_.inlinks;
  for (auto& item : inlinks) {
    const auto& var = item.second;
    const auto& tensor = var->Get<LoDTensor>();
    TensorArray& ta = step_inputs_[item.first];
    dy_seq_metas_[item.first] =
        ta.Unpack(tensor, level, true /*length_descend*/);

    if (cache_.num_steps) {
      PADDLE_ENFORCE_EQ(ta.size(), cache_.num_steps,
                        "inputs should have the same steps");
    } else {
      cache_.num_steps = ta.size();
    }
  }
}

void DynamicRecurrentOp::WriteStepInputs() const {
  const auto& inlinks = cache_.inlinks;
  for (auto& item : inlinks) {
    auto ta_it = step_inputs_.find(item.first);
    PADDLE_ENFORCE(ta_it != step_inputs_.end(), "");
    TensorArray& ta = step_inputs_[item.first];
    for (size_t step = 0; step < ta.size(); step++) {
      auto tensor = ta.Read(step);
      auto& step_scope = cache_.GetScope(step);
      auto var = step_scope.FindVar(item.first);
      PADDLE_ENFORCE_NOT_NULL(var);
      var->GetMutable<LoDTensor>()->ShareDataWith<value_type>(tensor);
    }
  }
}

void DynamicRecurrentOp::WriteStepOutputs() const {
  for (size_t step = 0; step < cache_.scopes->size(); step++) {
    auto& scope = cache_.GetScope(step);
    for (auto& item : step_outputs_) {
      const auto& step_output_var = scope.FindVar(item.first);
      auto& step_output = step_output_var->Get<LoDTensor>();
      item.second.WriteShared(step, step_output);
    }
  }
}

void DynamicRecurrentOp::CreateScopes() const {
  PADDLE_ENFORCE_GT(cache_.num_steps, 0);
  // resize scopes
  for (size_t i = 0; i < cache_.num_steps - cache_.scopes->size(); i++) {
    cache_.scopes->emplace_back(new Scope);
  }

  // init temporary inputs
  for (size_t step = cache_.scopes->size(); step < cache_.num_steps; step++) {
    auto& scope = cache_.GetScope(step);
    // create temporary inputs
    for (auto& input : stepnet_->Inputs()) {
      for (const std::string& var : input.second) {
        if (!scope.FindVar(var)) {
          scope.NewVar(var)->GetMutable<LoDTensor>();
        }
      }
    }

    // create temporary outputs
    for (auto& input : stepnet_->Outputs()) {
      for (const std::string& var : input.second) {
        if (!scope.FindVar(var)) {
          scope.NewVar(var)->GetMutable<LoDTensor>();
        }
      }
    }
  }
}

void DynamicRecurrentOp::ConcatOutputs() const {
  // TODO(superjom) transform this to a config
  int level = 0;
  // TODO(superjom) pass in some lod
  // just a placeholder
  framework::LoD lod;
  for (auto& item : step_outputs_) {
    auto tensor = item.second.Pack(level, dy_seq_metas_[item.first], lod);
    auto& output = cache_.outlinks[item.first]->Get<LoDTensor>();
    const_cast<LoDTensor*>(&output)->ShareDataWith<value_type>(tensor);
  }
}

void DynamicRecurrentOp::InitStates() const {
  // init the first state
  // TODO(superjom) parepare the scenerio that boot state not exists
  for (auto memory : arg_.memories) {
    auto* boot_state_var = cache_.scope->FindVar(memory.boot_var);
    PADDLE_ENFORCE_NOT_NULL(boot_state_var);
    auto& boot_state = boot_state_var->Get<LoDTensor>();
    for (size_t step = 0; step < cache_.num_steps; step++) {
      // link pre-state to boot_state
      auto& cur_scope = cache_.GetScope(step);
      auto* var = cur_scope.FindVar(memory.pre_var);
      PADDLE_ENFORCE_NOT_NULL(var);
      if (step == 0) {
        auto* cur_state_tensor = var->GetMutable<LoDTensor>();
        cur_state_tensor->Resize(boot_state.dims());
        cur_state_tensor->ShareDataWith<value_type>(boot_state);
      } else {
        auto& pre_scope = cache_.GetScope(step - 1);
        auto* state_pre = pre_scope.FindVar(memory.var);
        PADDLE_ENFORCE_NOT_NULL(state_pre);
        auto* pre_state = cur_scope.FindVar(memory.pre_var);
        pre_state->GetMutable<LoDTensor>()->ShareDataWith<value_type>(
            state_pre->Get<LoDTensor>());
      }
    }
  }
}

void DynamicRecurrentOp::ArgCache::Init(
    const rnn::ArgumentName& name, const paddle::framework::OperatorBase& op,
    const paddle::framework::Scope& scope, rnn::Argument* arg) {
  this->scope = &scope;
  InitArgument(name, op, arg);
  CacheScopes(scope, *arg);
  CacheInlinks(scope, arg->inlinks);
  CacheOutlinks(scope, arg->outlinks);
}

void DynamicRecurrentOp::ArgCache::InitArgument(const rnn::ArgumentName& name,
                                                const OperatorBase& op,
                                                rnn::Argument* arg) {
  rnn::InitArgument(name, arg, op, false /*is_grad*/);
}

void DynamicRecurrentOp::ArgCache::CacheScopes(const Scope& scope,
                                               const rnn::Argument& arg) {
  auto scopes_var = scope.FindVar(arg.step_scopes);
  PADDLE_ENFORCE(scopes_var != nullptr,
                 "the step_scopes output argument [%s] should be created first "
                 "by framework.",
                 arg.step_scopes);
  scopes = scopes_var->GetMutable<std::vector<std::unique_ptr<Scope>>>();
}

void DynamicRecurrentOp::ArgCache::CacheInlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    inlinks[name] = var;
  }
}

void DynamicRecurrentOp::ArgCache::CacheOutlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    outlinks[name] = var;
  }
}

Variable* DynamicRecurrentOp::ArgCache::GetVariable(const Scope& scope,
                                                    const std::string& name) {
  auto* var = scope.FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(var, "variable [%s] not exist in scope", name);
  return var;
}

const rnn::ArgumentName DynamicRecurrentOp::kArgName{
    "step_net", "step_scopes",  "inlinks",      "outlinks",
    "memories", "pre_memories", "boot_memories"};

void DynamicRecurrentGradientOp::Run(
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    dynamic_recurrent_op, paddle::operators::DynamicRecurrentOp,
    paddle::operators::DynamicRecurrentAlgorithmProtoAndCheckerMaker);
