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

namespace paddle {
namespace operators {

using namespace paddle::framework;

namespace rnn {

/**
 * Memory of a RNN (same as the role of `Momory` in PaddlePaddle).
 *
 * Memory attributes cached by this op, dims will be infered from
 * boot memories in father scope. Other attributes are copied from Op's proto
 * attributes.
 */
struct MemoryAttr {
  // name of current state variable
  std::string var;
  // name of previous step's state variable
  std::string pre_var;
  // name of the variables to init this memory (same role of `boot_layer` in
  // PaddlePaddle), which is store in father's scope.
  std::string boot_var;
};

struct Link {
  // input or output links name.
  std::string internal;
  // alias to avoid duplicate keys in scopes.
  std::string external;
};

struct Argument {
  std::string step_net;
  std::string step_scopes;
  std::vector<Link> inlinks;
  std::vector<Link> outlinks;
  std::vector<rnn::MemoryAttr> memories;
};

struct ArgumentName {
  std::string step_net;
  std::string step_scopes;
  std::string inlinks;
  std::string outlinks;
  std::string inlink_alias;   // the alias of inlinks in step net.
  std::string outlink_alias;  // the alias of outlinks in step net.
  std::string memories;       // the memory name
  std::string pre_memories;   // the previous memory name
  std::string boot_memories;  // the boot memory name
};

/**
 * Prepare inputs for each step net.
 */
void SegmentInputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<Link>& inlinks,
                   const size_t seq_len);

/**
 * Process outputs of step nets and merge to variables.
 */
void ConcatOutputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<Link>& outlinks,
                   const size_t seq_len);

void LinkMemories(const std::vector<Scope*>& step_scopes,
                  const std::vector<MemoryAttr>& memories,
                  size_t step_id,
                  int offset);

void InitArgument(const ArgumentName& name, Argument* arg);

};  // namespace rnn

// The sequence format in RecurrentOp is Tensor<seq_len, batch_size, dim> now.
// TODO:
// 1. No-padding computing for sequences with indifinite length in one batch.
// 2. Hierarchical RNN for sequence with sub-sequence.
// 3. Internal Memory.
// 4. More Complex RNN architecture, such as Gated Feedback RNN.
//    Refer to: https://arxiv.org/pdf/1502.02367.pdf

class RecurrentAlgorithm {
public:
  void Run(const Scope& scope, const platform::DeviceContext& dev_ctx) const;

  void Init(std::unique_ptr<rnn::Argument> arg) { arg_ = std::move(arg); }

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const Scope& scope) const;

protected:
  /*
   * The step scopes will be stored in the father scope as a variable.
   *
   * NOTE the scopes are reused in both the forward and backward, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(const Scope& scope) const;

  const std::vector<Scope*>& GetStepScopes(const Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)->GetMutable<std::vector<Scope*>>();
  }

  void InitMemories(Scope* step_scopes) const;

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

  void Run(const Scope& scope, const platform::DeviceContext& dev_ctx) const;

  void LinkBootMemoryGradients(Scope* step_scopes) const;

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const Scope& scope) const;

protected:
  inline const std::vector<Scope*>& GetStepScopes(const Scope& scope) const {
    return *scope.FindVar(arg_->step_scopes)->GetMutable<std::vector<Scope*>>();
  }

private:
  std::unique_ptr<rnn::Argument> arg_;
  mutable size_t seq_len_;
};

class RecurrentOp final : public OperatorBase {
public:
  void Init() override;

  /**
   * InferShape must be called before Run.
   */
  virtual void InferShape(const Scope& scope) const override {
    alg_.InferShape(scope);
  }

  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  static const rnn::ArgumentName kArgName;

private:
  RecurrentAlgorithm alg_;
};

class RecurrentGradientOp final : public OperatorBase {
public:
  void Init() override;

  /**
   * InferShape must be called before Run.
   */
  virtual void InferShape(const Scope& scope) const override {
    alg_.InferShape(scope);
  }

  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    alg_.Run(scope, dev_ctx);
  }

  static const rnn::ArgumentName kArgName;

private:
  RecurrentGradientAlgorithm alg_;
};

}  // namespace operators
}  // namespace paddle
