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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Scope;
using framework::TensorArray;
using framework::LoDTensor;
using framework::Variable;
using framework::OperatorBase;
using framework::DySeqMetaBatch;

namespace detail {

inline void CreateVariables(Scope& scope,
                            const std::vector<std::string>& var_names) {
  for (const auto& name : var_names) {
    scope.Var(name);
  }
}

template <typename T>
inline const LoDTensor& EigenAddto(LoDTensor* to, LoDTensor& from,
                                   Eigen::DefaultDevice place) {
  auto to_eigen = framework::EigenVector<T>::Flatten(*to);
  auto from_eigen = framework::EigenVector<T>::Flatten(from);
  to_eigen.device(place) += from_eigen;
  return *to;
}

rnn::StateAttr GenGradStateAttr(const rnn::StateAttr& state) {
  rnn::StateAttr res;
  PADDLE_ENFORCE(state.var.back() != 'D', "%s", state.var);
  // PADDLE_ENFORCE(state.boot_var.back() != 'D', "%s", state.boot_var);
  PADDLE_ENFORCE(state.pre_var.back() != 'D', "%s", state.pre_var);
  res.var = framework::GradVarName(state.var);
  res.pre_var = framework::GradVarName(state.pre_var);
  res.boot_var = state.boot_var;
  return res;
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
inline void ReorderInitialState(const DySeqMetaBatch& metas,
                                const LoDTensor& boot_state, LoDTensor* tensor,
                                const platform::Place& dst_place) {
  for (size_t seq_id = 0; seq_id < metas.size(); seq_id++) {
    auto slice = tensor->Slice(seq_id, seq_id + 1);
    auto boot_slice =
        boot_state.Slice(metas[seq_id].ori_idx, metas[seq_id].ori_idx + 1);
    // TODO(superjom) pass in device context as an argument
    slice.CopyFrom(boot_slice, dst_place, platform::CPUDeviceContext());
  }
}

inline void RestoreInitialState(const DySeqMetaBatch& metas,
                                const LoDTensor& tensor, LoDTensor* boot_state,
                                const platform::Place& dst_place) {
  for (size_t seq_id = 0; seq_id < metas.size(); seq_id++) {
    auto slice = tensor.Slice(seq_id, seq_id + 1);
    auto boot_slice =
        boot_state->Slice(metas[seq_id].ori_idx, metas[seq_id].ori_idx + 1);
    boot_slice.CopyFrom(slice, dst_place, platform::CPUDeviceContext());
  }
}

}  // namespace detail

// Implementation for forward propagation.
template <>
void RNNAlgorithm::Run<RNNAlgorithm::ComputeMode::kForward>(
    const framework::Scope& scope, const framework::OperatorBase& op,
    const platform::DeviceContext& dev_ctx) {
  SetComputeMode(ComputeMode::kForward);
  cache_.Init(kArgNames[mode_], op, scope, &dev_ctx, &arg_, false /*is_grad*/);
  SplitInputs();
  CreateScopes();
  WriteStepInputs();
  InitStates();
  WriteStepOutputs();
  RunSteps();
  ConcatOutputs();
}

// Implementation for backward propagation.
template <>
void RNNAlgorithm::Run<RNNAlgorithm::ComputeMode::kBackward>(
    const framework::Scope& scope, const framework::OperatorBase& op,
    const platform::DeviceContext& dev_ctx) {
  SetComputeMode(ComputeMode::kBackward);
  parameters_ = op.Outputs(framework::GradVarName("parameters"));
  cache_.Init(kArgNames[mode_], op, scope, &dev_ctx, &arg_, true /*is_grad*/);
  SplitInputs();
  CreateScopes();
  WriteStepInputs();
  InitStates();
  WriteStepOutputs();
  CreateLocalParameterGradients();
  RunSteps();
  // copy boot-states' gradients back.
  for (const auto& state : arg_.states) {
    ExportInitialStateGradient(state);
  }

  ExportWeightGradients();
  ConcatOutputs();
}

void RNNAlgorithm::SplitInputs() {
  // TODO(superjom) make level a config
  // TODO(superjom) check all the inputs has the same LoD
  int level = 0;
  for (const auto& item : cache_.inputs) {
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

void RNNAlgorithm::WriteStepInputs() {
  for (const auto& item : cache_.inputs) {
    auto ta_it = step_inputs_.find(item.first);
    PADDLE_ENFORCE(ta_it != step_inputs_.end(),
                   "step_inputs_ not compatible with memory set");
    TensorArray& ta = ta_it->second;
    for (size_t step = 0; step < ta.size(); step++) {
      auto tensor = ta.Read(step);
      auto& step_scope = cache_.GetScope(step);
      Variable* var = step_scope.FindVar(item.first);
      if (var == nullptr) {
        var = step_scope.Var(item.first);
      }
      var->GetMutable<LoDTensor>()->ShareDataWith(tensor);
    }
  }
}

void RNNAlgorithm::WriteStepOutputs() {
  // initialize step outputs
  for (const auto& item : cache_.outputs) {
    step_outputs_.emplace(item.first, TensorArray());
  }
  PADDLE_ENFORCE_GT(step_outputs_.size(), 0UL);
}

void RNNAlgorithm::CreateScopes() {
  PADDLE_ENFORCE_GT(cache_.num_steps, 0);
  // resize scopes
  size_t num_scopes_need_create = cache_.num_steps - cache_.scopes->size();
  for (size_t i = 0; i < num_scopes_need_create; i++) {
    cache_.scopes->emplace_back(&cache_.scope->NewScope());
  }

  // init temporary inputs
  PADDLE_ENFORCE_NOT_NULL(step_unit_, "stepnet should be set first");
  std::vector<std::string> states;
  std::vector<std::string> ex_states;
  std::vector<std::string> step_unit_outputs;
  std::transform(arg_.states.begin(), arg_.states.end(),
                 std::back_inserter(states), [this](const rnn::StateAttr& m) {
                   return IsForward() ? m.var : framework::GradVarName(m.var);
                 });
  std::transform(
      arg_.states.begin(), arg_.states.end(), std::back_inserter(ex_states),
      [this](const rnn::StateAttr& m) {
        return IsForward() ? m.pre_var : framework::GradVarName(m.pre_var);
      });
  for (const auto& item : step_unit_->Outputs()) {
    for (const auto& var : item.second) {
      step_unit_outputs.push_back(var);
    }
  }

  for (size_t step = 0; step < cache_.num_steps; step++) {
    auto& scope = cache_.GetScope(step);
    detail::CreateVariables(scope, arg_.inlinks);
    detail::CreateVariables(scope, arg_.outlinks);
    detail::CreateVariables(scope, states);
    detail::CreateVariables(scope, ex_states);
    detail::CreateVariables(scope, step_unit_outputs);
  }
}

void RNNAlgorithm::ConcatOutputs() {
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
  // the inputs' lods should be the same, so randomly get one lod.
  const auto& some_lod =
      cache_.scope->FindVar(arg_.inlinks.front())->Get<LoDTensor>().lod();
  const auto& some_meta = dy_seq_metas_[arg_.inlinks.front()];
  for (auto& item : step_outputs_) {
    auto tensor = item.second.Pack(level, some_meta, some_lod);
    auto* output = cache_.outputs[item.first]->GetMutable<LoDTensor>();
    const_cast<LoDTensor*>(output)->ShareDataWith(tensor);
  }
}

void RNNAlgorithm::RunSteps() {
  if (IsBackward()) {
    // call stepnet in all the time steps reversely
    for (int step = cache_.num_steps - 1; step >= 0; step--) {
      auto& step_scope = cache_.GetScope(step);
      // after run, the current state@grad will be assigned.
      step_unit_->Run(step_scope, *cache_.dev_ctx);
      // accumulate gradients to the state@grad in the previous scope.
      if (step < cache_.num_steps - 1) {
        for (auto& state : arg_.states) {
          // Add the state@pre@grad to the current state@grad
          LinkGradState(detail::GenGradStateAttr(state), step + 1);
        }
      }
    }
    // set the current state@grad to boot_state@grad
    for (auto& state : arg_.states) {
      LinkGradState(detail::GenGradStateAttr(state), 0);
    }
  } else {
    for (size_t step = 0; step < cache_.num_steps; step++) {
      auto& step_scope = cache_.GetScope(step);
      step_unit_->Run(step_scope, *cache_.dev_ctx);
    }
  }
}

void RNNAlgorithm::InitStates() {
  for (size_t step = 0; step < cache_.num_steps; step++) {
    for (const auto& state : arg_.states) {
      if (IsBackward()) {
        CreateState(detail::GenGradStateAttr(state), step);
      } else {
        // only forward need to link state before, the backward accumulate
        // `pre_state`'s grad during running time steps.
        CreateState(state, step);
        LinkState(state, step);
      }
    }
  }
}

void RNNAlgorithm::CreateState(const rnn::StateAttr& state_attr, size_t step) {
  auto& scope = cache_.GetScope(step);
  // if backward, init the LoDTensor
  // TODO(superjom) some improvement here
  if (IsForward() ||
      !scope.Var(state_attr.var)->IsType<framework::LoDTensor>()) {
    auto& state = *cache_.GetTensor(scope, state_attr.var);
    auto& boot_state = *cache_.GetTensor(*cache_.scope, state_attr.boot_var);

    size_t num_instances =
        step_inputs_[arg_.inlinks.front()].Read(step).dims()[0];
    auto dims = boot_state.dims();
    dims[0] = num_instances;

    state.Resize(dims);
    state.mutable_data<value_type>(platform::CPUPlace());
  }

  auto& state = *cache_.GetTensor(scope, state_attr.var);
  states_[state_attr.var].WriteShared(step, state);
}

void RNNAlgorithm::LinkState(const rnn::StateAttr& state, size_t step) {
  auto& scope = cache_.GetScope(step);
  auto& state_pre = *cache_.GetTensor(scope, state.pre_var);

  // process the first state's boot-state(the 0-step in forward mode or the
  // last step in backward mode)
  // Only forward mode need to link the boot-state to the `pre-state` in first
  // time step. In backward mode, need to copy the gradient of `pre-state` in
  // first time step to the gradient of `boot-state`.
  if (step == 0) {
    LinkInitialState(state);
  } else {
    size_t num_instances =
        step_inputs_[arg_.inlinks.front()].Read(step).dims()[0];
    auto* pre_state = cache_.GetTensor(cache_.GetScope(step - 1), state.var);
    // shink and share from previous state
    auto shrinked_pre_state = pre_state->Slice(0, num_instances);
    state_pre.ShareDataWith(shrinked_pre_state);
  }
  pre_states_[state.var].WriteShared(step, state_pre);
}

void RNNAlgorithm::LinkGradState(const rnn::StateAttr& state, size_t step) {
  if (step == 0) {
    ExportInitialStateGradient(state);
  } else {
    auto& scope = cache_.GetScope(step);
    auto& pre_scope = cache_.GetScope(step - 1);
    LoDTensor& pre_state = *cache_.GetTensor(scope, state.pre_var);
    LoDTensor& state_pre = *cache_.GetTensor(pre_scope, state.var);

    // TODO(superjom) to consider the more dims case
    size_t num_instances = pre_state.dims()[0];
    LoDTensor shrinked_state_pre;
    auto tmp_tensor = state_pre.Slice(0, num_instances);
    shrinked_state_pre.ShareDataWith(tmp_tensor);
    auto place = cache_.dev_ctx->GetEigenDevice<platform::CPUPlace>();
    detail::EigenAddto<value_type>(&shrinked_state_pre, pre_state, *place);
  }
}

void RNNAlgorithm::LinkInitialState(const rnn::StateAttr& state) {
  // all the step_inputs' metas should be the same, just randomly select one
  // and get the dyseq meta.
  const auto& some_meta = dy_seq_metas_[arg_.inlinks.front()];
  auto& scope = cache_.GetScope(0);
  auto& state_pre = *cache_.GetTensor(scope, state.pre_var);
  auto* pre_state = cache_.GetTensor(*cache_.scope, state.boot_var);
  pre_state->mutable_data<value_type>(platform::CPUPlace());
  // allocate state
  state_pre.Resize(pre_state->dims());
  state_pre.mutable_data<value_type>(platform::CPUPlace());
  detail::ReorderInitialState(some_meta, *pre_state, &state_pre,
                              pre_state->place());
}

void RNNAlgorithm::ExportInitialStateGradient(const rnn::StateAttr& state) {
  // all the step_inputs' metas should be the same, just randomly select one
  // and get the dyseq meta.
  const auto& some_meta = dy_seq_metas_[arg_.inlinks.front()];
  auto& scope = cache_.GetScope(0);

  auto& state_pre = *cache_.GetTensor(scope, state.pre_var);
  auto& pre_state = *cache_.GetTensor(*cache_.scope, state.boot_var);
  pre_state.Resize(state_pre.dims());
  pre_state.mutable_data<value_type>(platform::CPUPlace());
  detail::RestoreInitialState(some_meta, state_pre, &pre_state,
                              platform::CPUPlace());
}

void RNNAlgorithm::CreateLocalParameterGradients() {
  PADDLE_ENFORCE(!parameters_.empty(), "parameters should be set first");
  for (const auto& param_name : parameters_) {
    for (size_t step = 0; step < cache_.num_steps; step++) {
      cache_.GetScope(step).Var(param_name);
    }
  }
}

void RNNAlgorithm::ExportWeightGradients() {
  auto& parent_scope = *cache_.scope;
  // TODO(superjom) make place a customizable
  auto place = cache_.dev_ctx->GetEigenDevice<platform::CPUPlace>();
  for (const std::string& param : parameters_) {
    LoDTensor& param_gradient =
        *parent_scope.FindVar(param)->GetMutable<LoDTensor>();

    for (size_t step = 0; step < cache_.num_steps; step++) {
      LoDTensor& step_param = *cache_.GetTensor(cache_.GetScope(step), param);
      detail::EigenAddto<value_type>(&param_gradient, step_param, *place);
    }
  }
}

void RNNAlgorithm::ArgCache::Init(const rnn::ArgumentName& name,
                                  const paddle::framework::OperatorBase& op,
                                  const paddle::framework::Scope& scope,
                                  platform::DeviceContext const* dev_ctx,
                                  rnn::Argument* arg, bool is_grad = false) {
  this->scope = &scope;
  InitArgument(name, op, arg, is_grad);
  CacheScopes(scope, *arg);
  CacheInlinks(scope, arg->inlinks);
  CacheOutlinks(scope, arg->outlinks);
  this->dev_ctx = dev_ctx;
}

void RNNAlgorithm::ArgCache::InitArgument(const rnn::ArgumentName& name,
                                          const OperatorBase& op,
                                          rnn::Argument* arg,
                                          bool is_grad = false) {
  rnn::InitArgument(name, arg, op, is_grad);
}

void RNNAlgorithm::ArgCache::CacheScopes(const Scope& scope,
                                         const rnn::Argument& arg) {
  auto scopes_var = scope.FindVar(arg.step_scopes);
  PADDLE_ENFORCE_NOT_NULL(
      scopes_var,
      "the step_scopes output argument [%s] should be created first "
      "by framework.",
      arg.step_scopes);
  this->scopes = scopes_var->GetMutable<std::vector<Scope*>>();
}

void RNNAlgorithm::ArgCache::CacheInlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    inputs[name] = var;
  }
}

void RNNAlgorithm::ArgCache::CacheOutlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    outputs[name] = var;
  }
}

Variable* RNNAlgorithm::ArgCache::GetVariable(const Scope& scope,
                                              const std::string& name) {
  auto* var = scope.FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(var, "variable [%s] not exist in scope", name);
  return var;
}

LoDTensor* RNNAlgorithm::ArgCache::GetTensor(const framework::Scope& scope,
                                             const std::string& name) {
  auto* var = GetVariable(scope, name);
  return var->GetMutable<LoDTensor>();
}

const std::array<rnn::ArgumentName, 2> RNNAlgorithm::kArgNames{
    {rnn::ArgumentName{"step_unit", "step_scopes", "inputs", "outputs",
                       "states", "ex_states", "initial_states"},
     // NOTE the variable name step_scopes is unchanged between forward and
     // backward.
     rnn::ArgumentName{"step_unit", "step_scopes", "outputs@GRAD",
                       "inputs@GRAD", "states", "ex_states",
                       "initial_states@GRAD"}}};

void DynamicRecurrentOp::Run(const framework::Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  rnn.Run<RNNAlgorithm::ComputeMode::kForward>(
      scope, *dynamic_cast<const OperatorBase*>(this), dev_ctx);
}

void DynamicRecurrentGradientOp::Run(
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {
  rnn.Run<RNNAlgorithm::ComputeMode::kBackward>(
      scope, *dynamic_cast<const OperatorBase*>(this), dev_ctx);
}

class DynamicRecurrentOpProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  DynamicRecurrentOpProtoAndCheckerMaker(framework::OpProto* proto,
                                         framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name =
        RNNAlgorithm::kArgNames[RNNAlgorithm::ComputeMode::kForward];
    // inputs and outputs stored in proto
    AddInput(name.inlinks,
             "the inputs that need to be segmented for each step.")
        .AsDuplicable();
    AddInput(name.initial_states, "variables to initialize states.")
        .AsDuplicable();
    AddInput("parameters", "parameter variables").AsDuplicable();

    AddOutput(name.outlinks, "the outputs that need to concated for all steps.")
        .AsDuplicable();
    AddOutput(name.step_scopes, "step scopes");

    // Attributes stored in AttributeMap
    AddAttr<std::vector<std::string>>(name.ex_states, "names of ex_states");
    AddAttr<std::vector<std::string>>(name.states, "names of states");

    AddComment("This is a RNN operator for varience-length sequences.");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(dynamic_recurrent, paddle::operators::DynamicRecurrentOp,
            paddle::operators::DynamicRecurrentOpProtoAndCheckerMaker,
            dynamic_recurrent_grad,
            paddle::operators::DynamicRecurrentGradientOp);
