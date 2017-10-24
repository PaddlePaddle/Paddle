/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_TESTING
#include "gtest/gtest.h"
#endif

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor_array.h"
#include "paddle/framework/variable.h"
#include "paddle/operators/rnn/recurrent_op_utils.h"

namespace paddle {
namespace operators {

class RNNAlgorithm {
 public:
  enum ComputeMode { kForward = 0, kBackward = 1 };
  static const std::array<rnn::ArgumentName, 2> kArgNames;
  using value_type = float;

  /*
   * Different `Run` method for forward and backward, `_` is just for template
   * specifialization.
   */
  template <ComputeMode _>
  void Run(const framework::Scope& scope, const framework::OperatorBase& op,
           const platform::DeviceContext& dev_ctx);
  /*
   * Split the inputs(LoDTensors) to segments for each time step.
   */
  void SplitInputs();

  /*
   * Create step-scopes to store temporary outputs in each time steps.
   */
  void CreateScopes();

  /*
   * Link TensorArray steps to the corresponding variables located in
   * step-scopes.
   */
  void WriteStepInputs();

  /*
   * Write output of each step to the corresponding TensorArray.
   */
  void WriteStepOutputs();

  /*
   * Initialize the states, each state will have a corresponding pre-state,
   * which share the memory with the state in the previous time state. The
   * pre-state in the first time step will be initialized with an zero tensor or
   * a tensor in parent scope if is provided.
   */
  void InitStates();

  /*
   * Create state variables for each time step.
   */
  void CreateState(const rnn::StateAttr& state, size_t step);

  /*
   * Link pre-state variable in current scope to the state variable in the
   * previous time step (scope) by reference.
   */
  void LinkState(const rnn::StateAttr& state, size_t step);

  /*
   * Link the pre-state of the first time step to the `boot-state` in parent's
   * scope.
   */
  void LinkInitialState(const rnn::StateAttr& state);

  /*
   * Copy the gradient from `pre-state` in the first step-scope to the
   * `boot-state` in parent's scope.
   */
  void ExportInitialStateGradient(const rnn::StateAttr& state);

  /*
   * Calculate time steps.
   */
  void RunSteps();

  /*
   * Concatenate outputs in each time step and generate a LoDTensor.
   */
  void ConcatOutputs();

  void SetComputeMode(ComputeMode mode) { mode_ = mode; }
  bool IsForward() const { return mode_ == ComputeMode::kForward; }
  bool IsBackward() const { return mode_ == ComputeMode::kBackward; }

  /*
   * set a step unit that is created according to a RecurrentOp's step unit.
   */
  void SetStepUnit(std::unique_ptr<framework::OperatorBase> step_unit) {
    PADDLE_ENFORCE_NOT_NULL(step_unit);
    step_unit_ = std::move(step_unit);
  }
  const framework::OperatorBase& GetStepUnit() const { return *step_unit_; }

  const framework::TensorArray& state(const std::string& name) const {
    auto it = states_.find(name);
    PADDLE_ENFORCE(it != states_.end());
    return it->second;
  }
  const framework::TensorArray& step_input(const std::string& name) const {
    auto it = step_inputs_.find(name);
    PADDLE_ENFORCE(it != step_inputs_.end());
    return it->second;
  }
  const framework::TensorArray& step_output(const std::string& name) const {
    auto it = step_outputs_.find(name);
    PADDLE_ENFORCE(it != step_outputs_.end());
    return it->second;
  }

 protected:
  struct ArgCache {
    framework::Scope const* scope;
    std::vector<framework::Scope*>* scopes;
    std::map<std::string, framework::Variable*> inputs;
    std::map<std::string, framework::Variable*> outputs;
    platform::DeviceContext const* dev_ctx;

    size_t num_steps{0};

    void Init(const rnn::ArgumentName& name, const framework::OperatorBase& op,
              const framework::Scope& scope,
              platform::DeviceContext const* dev_ctx, rnn::Argument* arg);

    framework::Scope& GetScope(size_t index) {
      PADDLE_ENFORCE_LT(index, num_steps);
      return *scopes->at(index);
    }

    framework::LoDTensor* GetTensor(const framework::Scope& scope,
                                    const std::string& name);

   private:
    void InitArgument(const rnn::ArgumentName& name,
                      const framework::OperatorBase& op, rnn::Argument* arg);
    void CacheScopes(const framework::Scope& scope, const rnn::Argument& arg);
    void CacheInlinks(const framework::Scope& scope,
                      const std::vector<std::string>& names);
    void CacheOutlinks(const framework::Scope& scope,
                       const std::vector<std::string>& names);
    framework::Variable* GetVariable(const framework::Scope& scope,
                                     const std::string& name);
  };

 private:
  std::unique_ptr<framework::OperatorBase> step_unit_;
  std::map<std::string, framework::TensorArray> states_;
  std::map<std::string, framework::TensorArray> step_inputs_;
  std::map<std::string, framework::TensorArray> step_outputs_;
  std::map<std::string, std::vector<framework::DySeqMeta>> dy_seq_metas_;
  rnn::Argument arg_;
  ArgCache cache_;
  ComputeMode mode_{ComputeMode::kForward};

#ifdef PADDLE_WITH_TESTING
  // test forward
  friend class RNNAlgorithmTestHelper;
  FRIEND_TEST(RNNAlgorithmTestHelper, SplitInputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, CreateCache);
  FRIEND_TEST(RNNAlgorithmTestHelper, CreateScopes);
  FRIEND_TEST(RNNAlgorithmTestHelper, WriteStepInputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, WriteStepOutputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, InitStates);
  FRIEND_TEST(RNNAlgorithmTestHelper, ConcatOutputs);
// TODO(superjom) test backward
#endif
};

class DynamicRecurrentOp : public framework::OperatorBase {
 public:
  DynamicRecurrentOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  DynamicRecurrentOp(const DynamicRecurrentOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

  mutable RNNAlgorithm rnn;
};

class DynamicRecurrentGradientOp : public framework::OperatorBase {
 public:
  DynamicRecurrentGradientOp(const std::string& type,
                             const framework::VariableNameMap& inputs,
                             const framework::VariableNameMap& outputs,
                             const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  DynamicRecurrentGradientOp(const DynamicRecurrentGradientOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

  mutable RNNAlgorithm rnn;
};

}  // namespace operators
}  // namespace paddle
