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

class DynamicRecurrentOp : public framework::OperatorBase {
 public:
  static const rnn::ArgumentName kArgName;
  using value_type = float;

  DynamicRecurrentOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  DynamicRecurrentOp(const DynamicRecurrentOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    // TODO(yuyang18): Implement copy ctor well.
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

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
   * previous time step (scope).
   */
  void LinkState(const rnn::MemoryAttr& memory, size_t step) const;

  /*
   * Concatenate outputs in each time step and generate a LoDTensor.
   */
  void ConcatOutputs() const;

  /*
   * set a stepnet that is created according to a RecurrentOp's stepnet.
   */
  void SetStepNet(std::unique_ptr<OperatorBase> net) {
    PADDLE_ENFORCE_NOT_NULL(net);
    stepnet_ = std::move(net);
  }
  const OperatorBase& GetStepNet() const { return *stepnet_; }

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

    size_t num_steps{0};

    void Init(const rnn::ArgumentName& name, const OperatorBase& op,
              const framework::Scope& scope, rnn::Argument* arg);

    framework::Scope& GetScope(size_t index) {
      PADDLE_ENFORCE_LT(index, num_steps);
      return *scopes->at(index);
    }

    framework::LoDTensor* GetTensor(const framework::Scope& scope,
                                    const std::string& name);

   private:
    void InitArgument(const rnn::ArgumentName& name, const OperatorBase& op,
                      rnn::Argument* arg);
    void CacheScopes(const framework::Scope& scope, const rnn::Argument& arg);
    void CacheInlinks(const framework::Scope& scope,
                      const std::vector<std::string>& names);
    void CacheOutlinks(const framework::Scope& scope,
                       const std::vector<std::string>& names);
    framework::Variable* GetVariable(const framework::Scope& scope,
                                     const std::string& name);
  };

 private:
  std::unique_ptr<OperatorBase> stepnet_;
  mutable std::map<std::string, framework::TensorArray> states_;
  mutable std::map<std::string, framework::TensorArray> step_inputs_;
  mutable std::map<std::string, framework::TensorArray> step_outputs_;
  mutable std::map<std::string, std::vector<framework::DySeqMeta>>
      dy_seq_metas_;
  mutable rnn::Argument arg_;
  mutable ArgCache cache_;

#ifdef PADDLE_WITH_TESTING
  friend class DynamicRecurrentOpTestHelper;
  FRIEND_TEST(DynamicRecurrentOpTestHelper, SplitInputs);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, CreateCache);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, CreateScopes);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, WriteStepInputs);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, WriteStepOutputs);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, InitStates);
  FRIEND_TEST(DynamicRecurrentOpTestHelper, ConcatOutputs);
#endif
};

class DynamicRecurrentGradientOp : public framework::OperatorBase {
 public:
  DynamicRecurrentGradientOp(const std::string& type,
                             const framework::VariableNameMap& inputs,
                             const framework::VariableNameMap& outputs,
                             const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;
};

}  // namespace operators
}  // namespace paddle
