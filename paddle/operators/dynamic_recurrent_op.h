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

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor_array.h"
#include "paddle/framework/variable.h"
#include "paddle/operators/rnn/recurrent_op_utils.h"

namespace paddle {
namespace operators {

using framework::Scope;
using framework::TensorArray;
using framework::LoDTensor;
using framework::Variable;

class DynamicRecurrentOp : public framework::OperatorBase {
 public:
  static const rnn::ArgumentName kArgName;
  using value_type = float;

  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

  /*
   * Split the inputs(LoDTensors) to segments for each time step.
   */
  void SplitInputs(const Scope& scope) const;

  /*
   * Create step-scopes to store temporary outputs in each time steps.
   */
  void CreateScopes(const Scope& scope) const;

  /*
   * Link TensorArray steps to the corresponding variables located in
   * step-scopes.
   */
  void WriteStepInputs(const Scope& scope) const;

  /*
   * Write output of each step to the corresponding TensorArray.
   */
  void WriteStepOutputs(const Scope& scope) const;

  /*
   * Initialize the states, each state will have a corresponding pre-state,
   * which share the memory with the state in the previous time state. The
   * pre-state in the first time step will be initialized with an zero tensor or
   * a tensor in parent scope if is provided.
   */
  void InitStates(const Scope& scope) const;

  /*
   * Concatenate outputs in each time step and generate a LoDTensor.
   */
  void ConcatOutputs(const Scope& scope) const;

  /*
   * set a stepnet that is created according to a RecurrentOp's stepnet.
   */
  void SetStepNet(std::unique_ptr<OperatorBase> net) {
    stepnet_ = std::move(net);
  }
  const OperatorBase& GetStepNet() const { return *stepnet_; }

  /*
   * Create the temporary inputs of a step-net in a step-scope.
   */
  void CreateTempInputsInScope(Scope& scope) const;

  /*
   * Create the temporary outputs of a step-net in a step-scope.
   */
  void CreateTempOutputsInScope(Scope& scope) const;

 protected:
  struct ArgCache {
    std::vector<Scope*>* scopes;
    std::map<std::string, Variable*> inlinks;
    std::map<std::string, Variable*> outlinks;

    size_t num_steps;

    void Init(const rnn::ArgumentName& name, const OperatorBase& op,
              const Scope& scope, const rnn::Argument* arg);

    Scope& GetScope(size_t index) {
      PADDLE_ENFORCE_LT(index, scopes->size());
      return *scopes->at(index);
    }

   protected:
    void InitArgument(const rnn::ArgumentName& name, const OperatorBase& op,
                      const rnn::Argument* arg);
    void CacheScopes(const Scope& scope, const rnn::Argument& arg);
    void CacheInlinks(const Scope& scope,
                      const std::vector<std::string>& names);
    void CacheOutlinks(const Scope& scope,
                       const std::vector<std::string>& names);
    Variable* GetVariable(const Scope& scope, const std::string& name);
  };

 private:
  std::unique_ptr<OperatorBase> stepnet_;
  mutable TensorArray states_;
  mutable std::map<std::string, TensorArray> step_inputs_;
  mutable std::map<std::string, TensorArray> step_outputs_;
  mutable std::map<std::string, std::vector<framework::DySeqMeta>>
      dy_seq_metas_;
  rnn::Argument arg_;
  mutable ArgCache arg_cache_;
};

}  // namespace operators
}  // namespace paddle
