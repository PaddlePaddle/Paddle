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

#include <vector>
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {
constexpr char kInputs[] = "inputs";
constexpr char kInitialStates[] = "initial_states";
constexpr char kParameters[] = "parameters";
constexpr char kOutputs[] = "outputs";
constexpr char kStepScopes[] = "step_scopes";
constexpr char kExStates[] = "ex_states";
constexpr char kStates[] = "states";
constexpr char kStepNet[] = "step_net";
constexpr char kReverse[] = "reverse";
constexpr char kIsTrain[] = "is_train";

using StepScopeVar = std::vector<framework::Scope *>;

class StepScopes {
 public:
  StepScopes(const framework::Scope &parent, StepScopeVar *scopes,
             bool is_train, size_t seq_len)
      : counter_(0UL), scopes_(scopes), is_train_(is_train) {
    size_t num_step_scopes = is_train ? seq_len : 2;
    PADDLE_ENFORCE(scopes->empty());
    scopes->reserve(static_cast<size_t>(num_step_scopes));
    for (size_t i = 0; i < num_step_scopes; ++i) {
      scopes->emplace_back(&parent.NewScope());
    }
  }

  framework::Scope &NextStep() {
    size_t scope_id = counter_;
    counter_++;
    return GetScope(scope_id);
  }

  framework::Scope &GetScope(size_t scope_id) const {
    if (!is_train_) {
      scope_id %= 2;
    }
    PADDLE_ENFORCE_LT(scope_id, scopes_->size());
    return *(*scopes_)[scope_id];
  }

  framework::Scope &ExScope() { return GetScope(counter_ - 1); }

 private:
  size_t counter_;
  StepScopeVar *scopes_;
  bool is_train_;
};

class RecurrentOp : public framework::OperatorBase {
 public:
  RecurrentOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto seq_len = static_cast<size_t>(this->GetSequenceLength(scope));
    VLOG(3) << "Static RNN input sequence length = " << seq_len;
    StepScopes scopes = CreateStepScopes(scope, seq_len);
    auto reverse = Attr<bool>(kReverse);

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepNet);
    for (auto *var : block->AllVars()) {
      var->SetPersistable(true);  // do not drop any variable inside RNN.
    }
    auto *program = block->Program();

    for (size_t i = 0; i < seq_len; ++i) {
      size_t seq_offset = reverse ? i : seq_len - i - 1;
      auto &cur_scope = scopes.NextStep();

      for (auto &iname : Inputs(kInputs)) {
        // Link inputs
        LinkVarWithCallback(
            scope, iname, &cur_scope, iname,
            [&seq_offset](const framework::Tensor &outside,
                          framework::Tensor *inside) {
              inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
            });
      }

      if (i == 0) {
        // Link initialize states
        auto &init_states = Inputs(kInitialStates);
        auto &ex_states = Attr<std::vector<std::string>>(kExStates);
        PADDLE_ENFORCE_EQ(init_states.size(), ex_states.size());
        for (size_t i = 0; i < init_states.size(); ++i) {
          LinkVar(scope, init_states[i], &cur_scope, ex_states[i]);
        }
      } else {
        // Link ex states
        auto &ex_states = Attr<std::vector<std::string>>(kExStates);
        auto &states = Attr<std::vector<std::string>>(kStates);
        auto &ex_scope = scopes.ExScope();

        PADDLE_ENFORCE_EQ(ex_states.size(), states.size());
        for (size_t i = 0; i < ex_states.size(); ++i) {
          LinkVar(ex_scope, states[i], &cur_scope, ex_states[i]);
        }
      }
      // Every inputs are linked now, execute!
      executor.Run(*program, &cur_scope, block->ID());

      for (auto &oname : Outputs(kOutputs)) {
        auto *src_var = cur_scope.FindVar(oname);
        PADDLE_ENFORCE(src_var != nullptr);
        auto &src_tensor = src_var->Get<framework::LoDTensor>();

        auto *dst_var = scope.FindVar(oname);
        PADDLE_ENFORCE(dst_var != nullptr);
        auto &dst_tensor = dst_var->Get<framework::LoDTensor>();
        auto dst_out = dst_tensor.Slice(seq_offset, seq_offset + 1);
        // Explicit copy output since the local RNN scope can be destroyed
        // early.
        dst_out.CopyFrom(src_tensor, dev_ctx.GetPlace(), dev_ctx);
      }
    }
  }

 private:
  template <typename Callback>
  static void LinkVarWithCallback(const framework::Scope &src_scope,
                                  const std::string &src_var_name,
                                  framework::Scope *dst_scope,
                                  const std::string &dst_var_name,
                                  Callback callback) {
    auto *src_var = src_scope.FindVar(src_var_name);
    PADDLE_ENFORCE(src_var != nullptr);
    auto &src_tensor = src_var->Get<framework::LoDTensor>();

    auto *dst_var = dst_scope->Var(dst_var_name);
    auto *dst_tensor = dst_var->GetMutable<framework::LoDTensor>();
    callback(src_tensor, dst_tensor);
  }

  static void LinkVar(const framework::Scope &src_scope,
                      const std::string &src_var_name,
                      framework::Scope *dst_scope,
                      const std::string &dst_var_name) {
    LinkVarWithCallback(
        src_scope, src_var_name, dst_scope, dst_var_name,
        [](const framework::Tensor &src, framework::Tensor *dst) {
          dst->ShareDataWith(src);
        });
  }

  StepScopes CreateStepScopes(const framework::Scope &scope,
                              size_t seq_len) const {
    auto *var = scope.FindVar(Output(kStepScopes));
    PADDLE_ENFORCE(var != nullptr);
    return StepScopes(scope, var->GetMutable<StepScopeVar>(),
                      Attr<bool>(kIsTrain), seq_len);
  }

  int64_t GetSequenceLength(const framework::Scope &scope) const {
    // Dim format SEQ_LEN, BATCH_SIZE, ...
    int64_t seq_len = -1;
    auto &all_inputs = Inputs(kInputs);
    PADDLE_ENFORCE(!all_inputs.empty());
    for (auto &iname : all_inputs) {
      auto *var = scope.FindVar(iname);
      PADDLE_ENFORCE(var != nullptr);
      PADDLE_ENFORCE(var->IsType<framework::LoDTensor>());
      auto &dim = var->Get<framework::LoDTensor>().dims();
      if (seq_len == -1) {
        seq_len = dim[0];
      } else {
        PADDLE_ENFORCE_EQ(seq_len, dim[0]);
      }
    }
    return seq_len;
  }
};

class RecurrentOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecurrentOpProtoMaker(framework::OpProto *proto,
                        framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(kInputs, "rnn inputs").AsDuplicable();
    AddInput(kInitialStates, "rnn initial states").AsDuplicable();
    AddInput(kParameters, "").AsDuplicable();
    AddOutput(kOutputs, "").AsDuplicable();
    AddOutput(kStepScopes, "");
    AddAttr<std::vector<std::string>>(kExStates, "");
    AddAttr<std::vector<std::string>>(kStates, "");
    AddAttr<framework::BlockDescBind *>(kStepNet, "");
    AddAttr<bool>(kReverse, "").SetDefault(false);
    AddAttr<bool>(kIsTrain, "").SetDefault(true);
    AddComment("");
  }
};

}  // namespace operators
}  // namespace paddle
REGISTER_OPERATOR(recurrent, paddle::operators::RecurrentOp,
                  paddle::operators::RecurrentOpProtoMaker);
