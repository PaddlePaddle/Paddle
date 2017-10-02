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

#include "paddle/framework/operator.h"
#include "paddle/operators/net_op.h"
#include "paddle/operators/rnn/recurrent_op_utils.h"

namespace paddle {
namespace operators {

// The sequence format in RecurrentOp is Tensor<seq_len, batch_size, dim> now.
// TODO(Superjom)
// 1. No-padding computing for sequences with indifinite length in one batch.
// 2. Hierarchical RNN for sequence with sub-sequence.
// 3. Internal Memory.
// 4. More Complex RNN architecture, such as Gated Feedback RNN.
//    Refer to: https://arxiv.org/pdf/1502.02367.pdf

class RecurrentAlgorithm {
 public:
  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const;

  void Init(rnn::Argument* arg,
            std::unique_ptr<framework::OperatorBase>* stepnet) {
    PADDLE_ENFORCE_NOT_NULL(stepnet, "stepnet should be set before.");
    arg_ = arg;
    stepnet_ = stepnet;
  }

 protected:
  /*
   * The step scopes will be stored in the father scope as a variable.
   *
   * NOTE the scopes are reused in both the forward and backward, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(const framework::Scope& scope, size_t seq_len) const;

  const std::vector<framework::Scope*>& GetStepScopes(
      const framework::Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)
                ->GetMutable<std::vector<framework::Scope*>>();
  }

  void InitMemories(framework::Scope* step_scopes) const;

 private:
  std::unique_ptr<framework::OperatorBase>* stepnet_;
  rnn::Argument* arg_;
};

class RecurrentGradientAlgorithm {
  /**
   * RNN's backward alogorithm.
   *
   * To accelerate the development of RecurrentGradientOp, we decouple RNN's
   * algorithm and `OperatorBase`'s implementation, the former contains the core
   * implementation of a RNN, and will keep stable even if the framework changes
   * a
   * lot, and the latter is a wrapper acts like an dapter for it to make RNN an
   * operator.
   */
 public:
  void Init(rnn::Argument* arg,
            std::unique_ptr<framework::OperatorBase>* stepnet) {
    PADDLE_ENFORCE_NOT_NULL(stepnet, "stepnet should be set before.");
    arg_ = std::move(arg);
    stepnet_ = stepnet;
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const;

  void LinkBootMemoryGradients(framework::Scope* step_scopes) const;

 protected:
  inline const std::vector<framework::Scope*>& GetStepScopes(
      const framework::Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)
                ->GetMutable<std::vector<framework::Scope*>>();
  }

 private:
  rnn::Argument* arg_;
  std::unique_ptr<framework::OperatorBase>* stepnet_;
};

class RecurrentOp : public framework::OperatorBase {
 public:
  RecurrentOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs);

  RecurrentOp(const RecurrentOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    // TODO(yuyang18): Implement copy ctor well.
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  void set_stepnet(std::unique_ptr<OperatorBase> net) {
    stepnet_ = std::move(net);
  }

  const OperatorBase& stepnet() const { return *stepnet_; }

  static const rnn::ArgumentName kArgName;

 private:
  RecurrentAlgorithm alg_;
  rnn::Argument arg_;
  std::unique_ptr<OperatorBase> stepnet_;
};

class RecurrentGradientOp : public framework::OperatorBase {
 public:
  RecurrentGradientOp(const std::string& type,
                      const framework::VariableNameMap& inputs,
                      const framework::VariableNameMap& outputs,
                      const framework::AttributeMap& attrs);

  RecurrentGradientOp(const RecurrentGradientOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    // TODO(yuyang18): Implement Copy ctor.
    PADDLE_THROW("Not Implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  static const rnn::ArgumentName kArgName;

  /*
   * set a stepnet that is created according to a RecurrentOp's stepnet.
   */
  void set_stepnet(std::unique_ptr<OperatorBase> net) {
    stepnet_ = std::move(net);
  }
  const OperatorBase& stepnet() const { return *stepnet_; }

 private:
  RecurrentGradientAlgorithm alg_;
  std::unique_ptr<OperatorBase> stepnet_;
  rnn::Argument arg_;
};

}  // namespace operators
}  // namespace paddle
