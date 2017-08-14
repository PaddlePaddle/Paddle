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
#include "paddle/operators/rnn/recurrent_op_utils.h"

namespace paddle {
namespace operators {

// The sequence format in RecurrentOp is Tensor<seq_len, batch_size, dim> now.
// TODO(Yan Chunwei):
// 1. No-padding computing for sequences with indifinite length in one batch.
// 2. Hierarchical RNN for sequence with sub-sequence.
// 3. Internal Memory.
// 4. More Complex RNN architecture, such as Gated Feedback RNN.
//    Refer to: https://arxiv.org/pdf/1502.02367.pdf

class RecurrentAlgorithm {
 public:
  void Run(const framework::Scope& scope,
           platform::DeviceContext* dev_ctx) const;

  void Init(std::unique_ptr<rnn::Argument> arg) { arg_ = std::move(arg); }

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const framework::Scope& scope) const;

 protected:
  /*
   * The step scopes will be stored in the father scope as a variable.
   *
   * NOTE the scopes are reused in both the forward and backward, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(const framework::Scope& scope) const;

  const std::vector<framework::Scope*>& GetStepScopes(
      const framework::Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)
                ->GetMutable<std::vector<framework::Scope*>>();
  }

  void InitMemories(framework::Scope* step_scopes, bool infer_shape_mode) const;

 private:
  std::unique_ptr<rnn::Argument> arg_;
  mutable size_t seq_len_;
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
  void Init(std::unique_ptr<rnn::Argument> arg) { arg_ = std::move(arg); }

  void Run(const framework::Scope& scope,
           platform::DeviceContext* dev_ctx) const;

  void LinkBootMemoryGradients(framework::Scope* step_scopes,
                               bool infer_shape_mode) const;

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const framework::Scope& scope) const;

 protected:
  inline const std::vector<framework::Scope*>& GetStepScopes(
      const framework::Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)
                ->GetMutable<std::vector<framework::Scope*>>();
  }

 private:
  std::unique_ptr<rnn::Argument> arg_;
  mutable size_t seq_len_;
};

class RecurrentOp final : public framework::OperatorBase {
 public:
  RecurrentOp(const std::string& type, const VarNameMap& inputs,
              const VarNameMap& outputs, const framework::AttributeMap& attrs);
  /**
     * InferShape must be called before Run.
     */
  void InferShape(const framework::Scope& scope) const override {
    alg_.InferShape(scope);
  }

  void Run(const framework::Scope& scope,
           platform::DeviceContext* dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  static const rnn::ArgumentName kArgName;

 private:
  RecurrentAlgorithm alg_;
};

class RecurrentGradientOp final : public framework::OperatorBase {
 public:
  RecurrentGradientOp(const std::string& type, const VarNameMap& inputs,
                      const VarNameMap& outputs,
                      const framework::AttributeMap& attrs);

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const framework::Scope& scope) const override {
    alg_.InferShape(scope);
  }

  void Run(const framework::Scope& scope,
           platform::DeviceContext* dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  static const rnn::ArgumentName kArgName;

 private:
  RecurrentGradientAlgorithm alg_;
};

}  // namespace operators
}  // namespace paddle
