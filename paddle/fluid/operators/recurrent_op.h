/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// StepScopes manages the scopes inside Recurrent Op.
//
// if is_train = False, then
//   there are two scopes for the RNN and just support forward
// else
//   the len(scopes) == seq_len
//
// if is_backward = True, then
//   reversely access scopes, delete useless ex-scope
// else
//   access scopes from beginning to end
class StepScopes {
 public:
  StepScopes(const platform::DeviceContext &dev_ctx,
             const framework::Scope &parent,
             std::vector<framework::Scope *> *scopes, bool is_train,
             size_t seq_len, bool is_backward = false);

  // Get the current scope
  framework::Scope &CurScope();

  // Get the ex-scope, which is the scope in previous time step
  framework::Scope &ExScope();

  // Move to next time step when forwarding
  void ForwardNext();

  // Delete ex-scope after using it, then move to next time step when
  // backwarding
  void BackwardNext(const platform::DeviceContext &dev_ctx,
                    framework::Scope *parent_scope);

 private:
  framework::Scope &GetScope(size_t scope_id) const;

  size_t counter_;
  std::vector<framework::Scope *> *scopes_;
  bool is_train_;
  bool is_backward_;
};

// Base class for RecurrentOp/RecurrentGradOp
//    Some common protected functions for RecurrentOp/RecurrentGradOp
class RecurrentBase : public framework::OperatorBase {
 public:
  static const char kInputs[];
  static const char kInitialStates[];
  static const char kParameters[];
  static const char kOutputs[];
  static const char kStepScopes[];
  static const char kHasStates[];
  static const char kExStates[];
  static const char kStates[];
  static const char kStepBlock[];
  static const char kReverse[];
  static const char kIsTrain[];
  static const char kSkipEagerDeletionVars[];
  static const char kInputGrads[];
  static const char kOutputGrads[];
  static const char kParamGrads[];
  static const char kInitStateGrads[];

  RecurrentBase(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs);

 protected:
  // Get SequenceLength from Scope
  //   The sequence length is got from input tensor. The input tensor's
  //   dimension should be [SEQ_LEN, ..., ...]. The first of the tensor's shape
  //   is SEQ_LEN. The second of the tensor's shape could be the batch size or
  //   nested sequence length.
  int64_t GetSequenceLength(const framework::Scope &scope) const;

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.Var, dst_vars)):
  //   dst_tensor.ShareDataWith(src_tensor)
  static void LinkTensor(const framework::Scope &src_scope,
                         const std::vector<std::string> &src_vars,
                         framework::Scope *dst_scope,
                         const std::vector<std::string> &dst_vars);

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.Var, dst_vars)):
  //   callback(src_tensor, &dst_tensor)
  template <typename Callback>
  static void LinkTensorWithCallback(const framework::Scope &src_scope,
                                     const std::vector<std::string> &src_vars,
                                     framework::Scope *dst_scope,
                                     const std::vector<std::string> &dst_vars,
                                     Callback callback,
                                     bool is_backward = false) {
    PADDLE_ENFORCE_EQ(src_vars.size(), dst_vars.size(),
                      platform::errors::InvalidArgument(
                          "Sizes of source vars and destination vars are not "
                          "equal in LinkTensor."));
    for (size_t i = 0; i < dst_vars.size(); ++i) {
      VLOG(10) << "Link " << src_vars[i] << " to " << dst_vars[i];
      AccessTensor(src_scope, src_vars[i], dst_scope, dst_vars[i], callback,
                   is_backward);
    }
  }

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.FindVar, dst_vars)):
  //   callback(src_tensor, &dst_tensor)
  template <typename Callback>
  static void LinkTensorWithCallback(const framework::Scope &src_scope,
                                     const std::vector<std::string> &src_vars,
                                     const framework::Scope &dst_scope,
                                     const std::vector<std::string> &dst_vars,
                                     Callback callback,
                                     bool is_backward = false) {
    PADDLE_ENFORCE_EQ(src_vars.size(), dst_vars.size(),
                      platform::errors::InvalidArgument(
                          "Sizes of source vars and destination vars are not "
                          "equal in LinkTensor."));
    for (size_t i = 0; i < dst_vars.size(); ++i) {
      VLOG(10) << "Link " << src_vars[i] << " to " << dst_vars[i];
      AccessTensor(src_scope, src_vars[i], dst_scope, dst_vars[i], callback,
                   is_backward);
    }
  }

  // (seq_len, shape) -> return [seq_len] + list(shape)
  static framework::DDim PrependDims(size_t seq_len,
                                     const framework::DDim &src);

 private:
  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           framework::Scope *dst_scope,
                           const std::string &dst_var_name, Callback callback,
                           bool is_backward = false) {
    auto *src_var = src_scope.FindVar(src_var_name);
    if (is_backward && src_var == nullptr) {
      return;
    }
    PADDLE_ENFORCE_NOT_NULL(
        src_var, platform::errors::NotFound("Source variable %s is not found.",
                                            src_var_name));
    auto &src_tensor = src_var->Get<framework::LoDTensor>();

    auto *dst_var = dst_scope->Var(dst_var_name);
    auto *dst_tensor = dst_var->GetMutable<framework::LoDTensor>();
    callback(src_tensor, dst_tensor);
  }

  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           const framework::Scope &dst_scope,
                           const std::string &dst_var_name, Callback callback,
                           bool is_backward = false) {
    auto *dst_var = dst_scope.FindVar(dst_var_name);
    if (is_backward && dst_var == nullptr) {
      return;
    }
    auto *src_var = src_scope.FindVar(src_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        src_var, platform::errors::NotFound("Source variable %s is not found.",
                                            src_var_name));
    auto &src_tensor = src_var->Get<framework::LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(
        dst_var, platform::errors::NotFound(
                     "Destination variable %s is not found.", src_var_name));
    auto *dst_tensor = dst_var->GetMutable<framework::LoDTensor>();
    callback(src_tensor, dst_tensor);
  }
};

class RecurrentOp : public RecurrentBase {
 public:
  RecurrentOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs);

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override;

 private:
  StepScopes CreateStepScopes(const platform::DeviceContext &dev_ctx,
                              const framework::Scope &scope,
                              size_t seq_len) const;
};

class RecurrentGradOp : public RecurrentBase {
 public:
  RecurrentGradOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs);

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override;

  StepScopes CreateStepScopes(const platform::DeviceContext &dev_ctx,
                              const framework::Scope &scope,
                              size_t seq_len) const;

  std::unordered_set<std::string> List2Set(
      const std::vector<std::string> &list) const;

  std::unordered_set<std::string> LocalVarNames(
      const framework::Scope &scope) const;

  static std::vector<std::string> GradVarLists(
      const std::vector<std::string> &var_names);
};

}  // namespace operators
}  // namespace paddle
