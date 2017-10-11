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

using framework::Scope;
using framework::TensorArray;
using framework::LoDTensor;
using framework::Variable;
using framework::DySeqMetaBatch;

namespace detail {

inline void CreateVariables(Scope& scope,
                            const std::vector<std::string>& var_names) {
  for (const auto& name : var_names) {
    scope.NewVar(name);
  }
}

/*
 * The inputs with sequence should be reordered when they are split, so the
 * boot_states should be reordered in the same order.
 *
 * NOTE This may require that the `pre_state` of the first time step should just
 * copy the `boot_state` rather than reference it, for that the content should
 * be reordered, but the RNN op should not change the `boot_state` as an input
 * variable's content.
 */
template <typename T>
inline void ReorderBootState(const DySeqMetaBatch& metas,
                             const LoDTensor& boot_state, LoDTensor* tensor,
                             const platform::Place& dst_place) {
  for (size_t seq_id = 0; seq_id < metas.size(); seq_id++) {
    auto slice = tensor->Slice<T>(seq_id, seq_id + 1);
    auto boot_slice =
        boot_state.Slice<T>(metas[seq_id].ori_idx, metas[seq_id].ori_idx + 1);
    // slice.CopyFrom<typename T>(boot_slice, dst_place);
    slice.template CopyFrom<T>(boot_slice, dst_place);
  }
}

}  // namespace detail

class DynamicRecurrentOpProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  DynamicRecurrentOpProtoAndCheckerMaker(framework::OpProto* proto,
                                         framework::OpAttrChecker* op_checker)
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

    AddComment("This is a RNN operator for varience-length sequences.");
  }
};

void DynamicRecurrentOp::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  cache_.Init(kArgName, *this, scope, &arg_);
  SplitInputs();
  CreateScopes();
  WriteStepInputs();
  InitStates();
  WriteStepOutputs();

  // call stepnet in all the time steps
  for (size_t step = 0; step < cache_.num_steps; step++) {
    auto& step_scope = cache_.GetScope(step);
    stepnet_->Run(step_scope, dev_ctx);
  }

  ConcatOutputs();
}

void DynamicRecurrentOp::SplitInputs() const {
  // TODO(superjom) make level a config
  // TODO(superjom) check all the inputs has the same LoD
  int level = 0;
  for (const auto& item : cache_.inlinks) {
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
  for (const auto& item : cache_.inlinks) {
    auto ta_it = step_inputs_.find(item.first);
    PADDLE_ENFORCE(ta_it != step_inputs_.end(),
                   "step_inputs_ not compatible with memory set");
    TensorArray& ta = ta_it->second;
    for (size_t step = 0; step < ta.size(); step++) {
      auto tensor = ta.Read(step);
      auto& step_scope = cache_.GetScope(step);
      Variable* var = step_scope.FindVar(item.first);
      if (var == nullptr) {
        var = step_scope.NewVar(item.first);
      }
      var->GetMutable<LoDTensor>()->ShareDataWith<value_type>(tensor);
    }
  }
}

void DynamicRecurrentOp::WriteStepOutputs() const {
  // initialize step outputs
  for (const auto& item : cache_.outlinks) {
    step_outputs_.emplace(item.first, TensorArray());
  }
  PADDLE_ENFORCE_GT(step_outputs_.size(), 0UL);
}

void DynamicRecurrentOp::CreateScopes() const {
  PADDLE_ENFORCE_GT(cache_.num_steps, 0);
  // resize scopes
  size_t num_scopes_need_create = cache_.num_steps - cache_.scopes->size();
  for (size_t i = 0; i < num_scopes_need_create; i++) {
    cache_.scopes->emplace_back(&cache_.scope->NewScope());
  }

  // init temporary inputs
  PADDLE_ENFORCE_NOT_NULL(stepnet_, "stepnet should be set first");
  std::vector<std::string> memories;
  std::vector<std::string> pre_memories;
  std::vector<std::string> stepnet_outputs;
  std::transform(arg_.memories.begin(), arg_.memories.end(),
                 std::back_inserter(memories),
                 [](const rnn::MemoryAttr& m) { return m.var; });
  std::transform(arg_.memories.begin(), arg_.memories.end(),
                 std::back_inserter(pre_memories),
                 [](const rnn::MemoryAttr& m) { return m.pre_var; });
  for (const auto& item : stepnet_->Outputs()) {
    for (const auto& var : item.second) {
      stepnet_outputs.push_back(var);
    }
  }

  for (size_t step = 0; step < cache_.num_steps; step++) {
    auto& scope = cache_.GetScope(step);
    detail::CreateVariables(scope, arg_.inlinks);
    detail::CreateVariables(scope, arg_.outlinks);
    detail::CreateVariables(scope, memories);
    detail::CreateVariables(scope, pre_memories);
    detail::CreateVariables(scope, stepnet_outputs);
  }
}

void DynamicRecurrentOp::ConcatOutputs() const {
  // TODO(superjom) transform this to a config
  int level = 0;
  for (size_t step = 0; step < cache_.num_steps; step++) {
    auto& scope = cache_.GetScope(step);
    for (auto& item : step_outputs_) {
      auto* var = scope.FindVar(item.first);
      PADDLE_ENFORCE_NOT_NULL(var);
      auto* tensor = var->GetMutable<LoDTensor>();
      tensor->mutable_data<value_type>(platform::CPUPlace());
      item.second.WriteShared(step, *tensor);
    }
  }
  // the inlinks' lods should be the same, so randomly get one lod.
  const auto& some_lod =
      cache_.scope->FindVar(arg_.inlinks.front())->Get<LoDTensor>().lod();
  const auto& some_meta = dy_seq_metas_[arg_.inlinks.front()];
  for (auto& item : step_outputs_) {
    auto tensor = item.second.Pack(level, some_meta, some_lod);
    auto* output = cache_.outlinks[item.first]->GetMutable<LoDTensor>();
    const_cast<LoDTensor*>(output)->ShareDataWith<value_type>(tensor);
  }
}

void DynamicRecurrentOp::InitStates() const {
  for (size_t step = 0; step < cache_.num_steps; step++) {
    for (const auto& memory : arg_.memories) {
      CreateState(memory, step);
      LinkState(memory, step);
    }
  }
}

void DynamicRecurrentOp::CreateState(const rnn::MemoryAttr& memory,
                                     size_t step) const {
  auto& scope = cache_.GetScope(step);
  auto& state = *cache_.GetTensor(scope, memory.var);
  auto& boot_state = *cache_.GetTensor(*cache_.scope, memory.boot_var);

  size_t num_instances =
      step_inputs_[arg_.inlinks.front()].Read(step).dims()[0];
  auto dims = boot_state.dims();
  dims[0] = num_instances;

  state.Resize(dims);
  state.mutable_data<value_type>(platform::CPUPlace());
  states_[memory.var].WriteShared(step, state);
}

void DynamicRecurrentOp::LinkState(const rnn::MemoryAttr& memory,
                                   size_t step) const {
  auto& scope = cache_.GetScope(step);
  auto& state_pre = *cache_.GetTensor(scope, memory.pre_var);

  // all the step_inputs' metas should be the same, just randomly select one
  // and get the dyseq meta.
  const auto& some_meta = dy_seq_metas_[arg_.inlinks.front()];
  size_t num_instances =
      step_inputs_[arg_.inlinks.front()].Read(step).dims()[0];

  LoDTensor* pre_state{nullptr};
  if (step == 0) {
    pre_state = cache_.GetTensor(*cache_.scope, memory.boot_var);
    pre_state->mutable_data<float>(platform::CPUPlace());
    // allocate memory
    state_pre.Resize(pre_state->dims());
    state_pre.mutable_data<value_type>(platform::CPUPlace());
    detail::ReorderBootState<value_type>(some_meta, *pre_state, &state_pre,
                                         pre_state->place());
  } else {
    pre_state = cache_.GetTensor(cache_.GetScope(step - 1), memory.var);
  }

  // shink and share from previous state
  auto shrinked_pre_state = pre_state->Slice<value_type>(0, num_instances);
  state_pre.ShareDataWith<value_type>(shrinked_pre_state);
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
  this->scopes = scopes_var->GetMutable<std::vector<Scope*>>();
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

LoDTensor* DynamicRecurrentOp::ArgCache::GetTensor(
    const framework::Scope& scope, const std::string& name) {
  auto* var = GetVariable(scope, name);
  return var->GetMutable<LoDTensor>();
}

const rnn::ArgumentName DynamicRecurrentOp::kArgName{
    "step_net", "step_scopes",  "inlinks",      "outlinks",
    "memories", "pre_memories", "boot_memories"};

void DynamicRecurrentGradientOp::Run(
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    dynamic_recurrent, paddle::operators::DynamicRecurrentOp,
    paddle::operators::DynamicRecurrentOpProtoAndCheckerMaker);
