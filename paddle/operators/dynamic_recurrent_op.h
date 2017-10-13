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
   * different `Run` method for forward and backward.
   */
  template <ComputeMode _>
  void Run(const framework::Scope& scope, const framework::OperatorBase& op,
           const platform::DeviceContext& dev_ctx) const;
  /*
   * Split the inputs(LoDTensors) to segments for each time step.
   */
  void SplitInputs() const;

  /*
   * Create step-scopes to store temporary outputs in each time steps.
   */
  void CreateScopes() const;

  /*
   * Link TensorArray steps to the corresponding variables located in
   * step-scopes.
   */
  void WriteStepInputs() const;

  /*
   * Write output of each step to the corresponding TensorArray.
   */
  void WriteStepOutputs() const;

  /*
   * Initialize the states, each state will have a corresponding pre-state,
   * which share the memory with the state in the previous time state. The
   * pre-state in the first time step will be initialized with an zero tensor or
   * a tensor in parent scope if is provided.
   */
  void InitStates() const;

  /*
   * Create state variables for each time step.
   */
  void CreateState(const rnn::MemoryAttr& memory, size_t step) const;

  /*
   * Link pre-state variable in current scope to the state variable in the
   * previous time step (scope) by reference.
   */
  void LinkState(const rnn::MemoryAttr& memory, size_t step) const;

  /*
   * Link the pre-state of the first time step to the `boot-state` in parent's
   * scope.
   */
  void LinkBootState(const rnn::MemoryAttr& memory) const;

  /*
   * Copy the gradient from `pre-state` in the first step-scope to the
   * `boot-state` in parent's scope.
   */
  void ExportBootStateGradient(const rnn::MemoryAttr& memory) const;

  /*
   * Calculate time steps.
   */
  void RunSteps() const;

  /*
   * Concatenate outputs in each time step and generate a LoDTensor.
   */
  void ConcatOutputs() const;

  void SetComputeMode(ComputeMode mode) const { mode_ = mode; }
  bool IsForward() const { return mode_ == ComputeMode::kForward; }
  bool IsBackward() const { return mode_ == ComputeMode::kBackward; }

  /*
   * set a stepnet that is created according to a RecurrentOp's stepnet.
   */
  void SetStepNet(std::unique_ptr<framework::OperatorBase> net) {
    PADDLE_ENFORCE_NOT_NULL(net);
    stepnet_ = std::move(net);
  }
  const framework::OperatorBase& GetStepNet() const { return *stepnet_; }

  const framework::TensorArray& state(const std::string& name) const {
    return states_[name];
  }
  const framework::TensorArray& step_input(const std::string& name) const {
    return step_inputs_[name];
  }
  const framework::TensorArray& step_output(const std::string& name) const {
    return step_outputs_[name];
  }

 protected:
  struct ArgCache {
    framework::Scope const* scope;
    std::vector<framework::Scope*>* scopes;
    std::map<std::string, framework::Variable*> inlinks;
    std::map<std::string, framework::Variable*> outlinks;
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
  std::unique_ptr<framework::OperatorBase> stepnet_;
  mutable std::map<std::string, framework::TensorArray> states_;
  mutable std::map<std::string, framework::TensorArray> step_inputs_;
  mutable std::map<std::string, framework::TensorArray> step_outputs_;
  mutable std::map<std::string, std::vector<framework::DySeqMeta>>
      dy_seq_metas_;
  mutable rnn::Argument arg_;
  mutable ArgCache cache_;
  mutable ComputeMode mode_{ComputeMode::kForward};

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

  RNNAlgorithm rnn;
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

 private:
  RNNAlgorithm rnn;
};

}  // namespace operators
}  // namespace paddle
