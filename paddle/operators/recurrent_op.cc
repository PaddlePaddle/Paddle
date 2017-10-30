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
#define GRAD_SUFFIX "@GRAD"
constexpr char kInputGrads[] = "inputs" GRAD_SUFFIX;
constexpr char kOutputGrads[] = "outputs" GRAD_SUFFIX;
constexpr char kParamGrads[] = "parameters" GRAD_SUFFIX;
constexpr char kInitStateGrads[] = "initial_states" GRAD_SUFFIX;

using StepScopeVar = std::vector<framework::Scope *>;

class StepScopes {
 public:
  StepScopes(const framework::Scope &parent, StepScopeVar *scopes,
             bool is_train, size_t seq_len, bool is_backward = false)
      : counter_(is_backward ? seq_len - 1 : 0UL),
        scopes_(scopes),
        is_train_(is_train),
        is_backward_(is_backward) {
    size_t num_step_scopes = is_train ? seq_len : 2;
    if (!is_backward_) {
      PADDLE_ENFORCE(scopes->empty());
      scopes->reserve(static_cast<size_t>(num_step_scopes));
      for (size_t i = 0; i < num_step_scopes; ++i) {
        scopes->emplace_back(&parent.NewScope());
      }
    }
  }

  framework::Scope &CurScope() { return GetScope(counter_); }

  framework::Scope &GetScope(size_t scope_id) const {
    if (!is_train_) {
      scope_id %= 2;
    }
    PADDLE_ENFORCE_LT(scope_id, scopes_->size());
    return *(*scopes_)[scope_id];
  }

  framework::Scope &ExScope() {
    auto &scope = GetScope(is_backward_ ? counter_ + 1 : counter_ - 1);
    return scope;
  }

  void Next() {
    if (is_backward_) {
      --counter_;
    } else {
      ++counter_;
    }
  }

 private:
  size_t counter_;
  StepScopeVar *scopes_;
  bool is_train_;
  bool is_backward_;
};

class RecurrentBase : public framework::OperatorBase {
 public:
  RecurrentBase(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 protected:
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
        [&src_var_name, &dst_var_name](const framework::Tensor &src,
                                       framework::Tensor *dst) {
          VLOG(4) << "Linking from " << src_var_name << " with shape("
                  << src.dims() << ") to " << dst_var_name;
          dst->ShareDataWith(src);
        });
  }

  static framework::DDim AppendSeqLenToDim(size_t seq_len,
                                           const framework::DDim &src) {
    auto dims = framework::vectorize(src);
    dims.insert(dims.begin(), static_cast<int64_t>(seq_len));
    return framework::make_ddim(dims);
  }
};

class RecurrentOp : public RecurrentBase {
 public:
  RecurrentOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : RecurrentBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto seq_len = static_cast<size_t>(this->GetSequenceLength(scope));
    VLOG(3) << "Static RNN input sequence length = " << seq_len;
    StepScopes scopes = CreateStepScopes(scope, seq_len);
    auto reverse = Attr<bool>(kReverse);

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepNet);
    auto *program = block->Program();

    for (size_t i = 0; i < seq_len; ++i) {
      size_t seq_offset = reverse ? seq_len - i - 1 : i;
      VLOG(3) << "Recurrent operate at the time step " << seq_offset;

      auto &cur_scope = scopes.CurScope();

      for (auto &iname : Inputs(kInputs)) {
        // Link inputs
        LinkVarWithCallback(
            scope, iname, &cur_scope, iname,
            [&seq_offset, &iname](const framework::Tensor &outside,
                                  framework::Tensor *inside) {
              inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
              auto dims = framework::vectorize(inside->dims());
              dims.erase(dims.begin());
              inside->Resize(framework::make_ddim(dims));
              VLOG(4) << "RNN input=" << iname << " shape=" << inside->dims();
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
      executor.Run(*program, &cur_scope, block->ID(),
                   false /*create_local_scope*/);

      for (auto &oname : Outputs(kOutputs)) {
        auto *src_var = cur_scope.FindVar(oname);
        PADDLE_ENFORCE(src_var != nullptr);
        auto &src_tensor = src_var->Get<framework::LoDTensor>();

        auto *dst_var = scope.FindVar(oname);
        PADDLE_ENFORCE(dst_var != nullptr);
        auto &dst_tensor = *dst_var->GetMutable<framework::LoDTensor>();

        if (i == 0) {  // create output tensor at begin
          dst_tensor.Resize(AppendSeqLenToDim(seq_len, src_tensor.dims()));
          dst_tensor.mutable_data(dev_ctx.GetPlace(), src_tensor.type());
        }

        auto dst_out = dst_tensor.Slice(seq_offset, seq_offset + 1);
        // Explicit copy output since the local RNN scope can be destroyed
        // early.
        dst_out.CopyFrom(src_tensor, dev_ctx.GetPlace(), dev_ctx);
      }
      scopes.Next();
    }
  }

 private:
  StepScopes CreateStepScopes(const framework::Scope &scope,
                              size_t seq_len) const {
    auto *var = scope.FindVar(Output(kStepScopes));
    PADDLE_ENFORCE(var != nullptr);
    return StepScopes(scope, var->GetMutable<StepScopeVar>(),
                      Attr<bool>(kIsTrain), seq_len);
  }
};

class RecurrentGradOp : public RecurrentBase {
 public:
  RecurrentGradOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : RecurrentBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto seq_len = static_cast<size_t>(GetSequenceLength(scope));
    StepScopes scopes = CreateStepScopes(scope, seq_len);
    auto reverse = Attr<bool>(kReverse);

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepNet);
    auto *program = block->Program();

    for (size_t step_id = 0; step_id < seq_len; ++step_id) {
      size_t seq_offset = reverse ? step_id : seq_len - step_id - 1;
      VLOG(3) << "Recurrent backward operate at the time step " << seq_offset;
      auto &cur_scope = scopes.CurScope();
      // Link OG for RNN
      for (auto &og_name : Inputs(kOutputGrads)) {
        LinkVarWithCallback(
            scope, og_name, &cur_scope, og_name,
            [&seq_offset, &og_name](const framework::Tensor &outside,
                                    framework::Tensor *inside) {
              inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
              auto dims = framework::vectorize(inside->dims());
              dims.erase(dims.begin());
              inside->Resize(framework::make_ddim(dims));
              VLOG(4) << "RNN OG.name=" << og_name
                      << " shape=" << inside->dims();
            });
      }

      // Link memories
      if (step_id != 0) {  // not at beginning
        auto &ex_states = Attr<std::vector<std::string>>(kExStates);
        auto &states = Attr<std::vector<std::string>>(kStates);
        auto &ex_scope = scopes.ExScope();
        PADDLE_ENFORCE_EQ(ex_states.size(), states.size());
        for (size_t i = 0; i < ex_states.size(); ++i) {
          auto ex_state_grad = framework::GradVarName(ex_states[i]);
          auto state_grad = framework::GradVarName(states[i]);
          LinkVar(ex_scope, ex_state_grad, &cur_scope, state_grad);
        }
      }

      // Run
      executor.Run(*program, &cur_scope, block->ID(),
                   false /*create_local_scope*/);

      auto local_var_names = LocalVarNames(cur_scope);

      // Accumulate params
      {
        auto &pg_names = Outputs(kParamGrads);
        auto &p_names = Inputs(kParameters);
        PADDLE_ENFORCE_EQ(pg_names.size(), p_names.size());

        for (size_t prog_id = 0; prog_id < pg_names.size(); ++prog_id) {
          auto param_grad_name = framework::GradVarName(p_names[prog_id]);
          if (local_var_names.find(param_grad_name) !=
              local_var_names.end()) {  // local scope contains that gradient
            if (step_id == 0) {         // zero outside tensor
              auto &inside_tensor = cur_scope.FindVar(param_grad_name)
                                        ->Get<framework::LoDTensor>();
              framework::AttributeMap attrs;

              attrs["data_type"] = framework::ToDataType(inside_tensor.type());
              attrs["shape"] = framework::vectorize2int(inside_tensor.dims());
              attrs["value"] = 0.0f;

              auto zero_op = framework::OpRegistry::CreateOp(
                  "fill_constant", {}, {{"Out", {pg_names[prog_id]}}}, attrs);
              zero_op->Run(scope, dev_ctx);
            }

            auto *outside_var = scope.FindVar(pg_names[prog_id]);
            PADDLE_ENFORCE(outside_var != nullptr);
            auto &outside_tensor =
                *outside_var->GetMutable<framework::LoDTensor>();

            std::string result_var_name;
            auto *local_result_var = cur_scope.Var(&result_var_name);
            auto &local_result_tensor =
                *local_result_var->GetMutable<framework::LoDTensor>();

            local_result_tensor.ShareDataWith(outside_tensor);

            auto sum_op = framework::OpRegistry::CreateOp(
                "sum", {{"X", {result_var_name, param_grad_name}}},
                {{"Out", {result_var_name}}}, {});
            sum_op->Run(cur_scope, dev_ctx);
          }
        }
      }

      // Copy IG
      {
        auto &inames = Inputs(kInputs);
        auto &ig_names = Outputs(kInputGrads);
        PADDLE_ENFORCE_EQ(inames.size(), ig_names.size());
        for (size_t i = 0; i < inames.size(); ++i) {
          auto &iname = inames[i];
          auto ig_name_inside = framework::GradVarName(iname);
          auto &ig_name_outside = ig_names[i];

          auto *outside = scope.FindVar(ig_name_outside)
                              ->GetMutable<framework::LoDTensor>();
          auto &inside =
              cur_scope.FindVar(ig_name_inside)->Get<framework::LoDTensor>();
          if (step_id == 0) {  // alloc memory
            outside->Resize(AppendSeqLenToDim(seq_len, inside.dims()));
            outside->mutable_data(dev_ctx.GetPlace(), inside.type());
          }

          auto dst = outside->Slice(seq_offset, seq_offset + 1);
          dst.CopyFrom(inside, dev_ctx.GetPlace(), dev_ctx);
        }
      }

      if (step_id + 1 == seq_len) {  // at_end
        // Copy Memory
        auto &init_state_grads = Outputs(kInitStateGrads);
        auto &ex_states = Attr<std::vector<std::string>>(kExStates);
        PADDLE_ENFORCE_EQ(init_state_grads.size(), ex_states.size());
      }
      scopes.Next();
    }
  }

 private:
  StepScopes CreateStepScopes(const framework::Scope &scope,
                              size_t seq_len) const {
    auto *var = scope.FindVar(Input(kStepScopes));
    PADDLE_ENFORCE(var != nullptr);
    return StepScopes(scope, var->GetMutable<StepScopeVar>(),
                      Attr<bool>(kIsTrain), seq_len, true /*is_backward*/);
  }

  std::unordered_set<std::string> LocalVarNames(
      const framework::Scope &scope) const {
    auto local_var_name_list = scope.GetAllNames(false);
    std::unordered_set<std::string> local_var_name_set;
    local_var_name_set.reserve(local_var_name_list.size());
    for (auto &each : local_var_name_list) {
      local_var_name_set.insert(each);
    }
    return local_var_name_set;
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

class RecurrentGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;
  using OpDescBind = framework::OpDescBind;

 protected:
  virtual std::unique_ptr<OpDescBind> Apply() const {
    auto *grad = new OpDescBind();
    grad->SetType(this->GradOpType());

    for (auto &input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param));
    }

    for (auto &output_param : this->OutputNames()) {
      if (output_param == "step_scopes") {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->Output(output_param));
      } else {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->OutputGrad(output_param));
      }
    }

    grad->SetAttrMap(this->Attrs());

    return std::unique_ptr<OpDescBind>(grad);
  }

  virtual std::string GradOpType() const {
    return this->ForwardOpType() + "_grad";
  }
};

}  // namespace operators
}  // namespace paddle
REGISTER_OPERATOR(recurrent, paddle::operators::RecurrentOp,
                  paddle::operators::RecurrentOpProtoMaker);
