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
constexpr char kStepBlock[] = "step_block";
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

  static void ShareVars(const framework::Scope &src_scope,
                        const std::vector<std::string> &src_vars,
                        framework::Scope *dst_scope,
                        const std::vector<std::string> &dst_vars) {
    PADDLE_ENFORCE_EQ(src_vars.size(), dst_vars.size());
    for (size_t i = 0; i < dst_vars.size(); ++i) {
      AccessTensor(src_scope, src_vars[i], dst_scope, dst_vars[i],
                   [&](const framework::Tensor &src, framework::Tensor *dst) {
                     VLOG(4) << "Linking from " << src_vars[i] << " with shape("
                             << src.dims() << ") to " << dst_vars[i];
                     dst->ShareDataWith(src);
                   });
    }
  }

  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::vector<std::string> &src_vars,
                           framework::Scope *dst_scope,
                           const std::vector<std::string> &dst_vars,
                           Callback callback) {
    PADDLE_ENFORCE_EQ(src_vars.size(), dst_vars.size());
    for (size_t i = 0; i < dst_vars.size(); ++i) {
      AccessTensor(src_scope, src_vars[i], dst_scope, dst_vars[i], callback);
    }
  }

  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::vector<std::string> &src_vars,
                           const framework::Scope &dst_scope,
                           const std::vector<std::string> &dst_vars,
                           Callback callback) {
    PADDLE_ENFORCE_EQ(src_vars.size(), dst_vars.size());
    for (size_t i = 0; i < dst_vars.size(); ++i) {
      AccessTensor(src_scope, src_vars[i], dst_scope, dst_vars[i], callback);
    }
  }

  static framework::DDim AppendSeqLenToDim(size_t seq_len,
                                           const framework::DDim &src) {
    auto dims = framework::vectorize(src);
    dims.insert(dims.begin(), static_cast<int64_t>(seq_len));
    return framework::make_ddim(dims);
  }

 private:
  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           framework::Scope *dst_scope,
                           const std::string &dst_var_name, Callback callback) {
    auto *src_var = src_scope.FindVar(src_var_name);
    PADDLE_ENFORCE(src_var != nullptr);
    auto &src_tensor = src_var->Get<framework::LoDTensor>();

    auto *dst_var = dst_scope->Var(dst_var_name);
    auto *dst_tensor = dst_var->GetMutable<framework::LoDTensor>();
    callback(src_tensor, dst_tensor);
  }

  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           const framework::Scope &dst_scope,
                           const std::string &dst_var_name, Callback callback) {
    auto *src_var = src_scope.FindVar(src_var_name);
    PADDLE_ENFORCE(src_var != nullptr);
    auto &src_tensor = src_var->Get<framework::LoDTensor>();
    auto *dst_var = dst_scope.FindVar(dst_var_name);
    PADDLE_ENFORCE(dst_var != nullptr);
    auto *dst_tensor = dst_var->GetMutable<framework::LoDTensor>();
    callback(src_tensor, dst_tensor);
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
    auto *block = Attr<framework::BlockDescBind *>(kStepBlock);
    auto *program = block->Program();

    for (size_t i = 0; i < seq_len; ++i) {
      size_t seq_offset = reverse ? seq_len - i - 1 : i;
      VLOG(3) << "Recurrent operate at the time step " << seq_offset;

      auto &cur_scope = scopes.CurScope();

      // Link outside::input --> inside::input
      //   inside::input = outside::input[seq_offset: seq_offset+1]
      AccessTensor(
          scope, Inputs(kInputs), &cur_scope, Inputs(kInputs),
          [&seq_offset](const framework::Tensor &outside,
                        framework::Tensor *inside) {
            inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
            auto dims = framework::vectorize(inside->dims());
            dims.erase(dims.begin());
            inside->Resize(framework::make_ddim(dims));
          });

      if (i == 0) {
        // Link initial states  --> ex_states
        ShareVars(scope, Inputs(kInitialStates), &cur_scope,
                  Attr<std::vector<std::string>>(kExStates));
      } else {
        auto &ex_scope = scopes.ExScope();
        // Link ex_scope::state --> cur_scope::ex_state
        ShareVars(ex_scope, Attr<std::vector<std::string>>(kStates), &cur_scope,
                  Attr<std::vector<std::string>>(kExStates));
      }

      // Every inputs are linked now, execute!
      executor.Run(*program, &cur_scope, block->ID(),
                   false /*create_local_scope*/);

      // Copy inside::output -> outside::output
      //    outside::output[seq_offset: seq_offset + 1] = inside::output
      this->AccessTensor(
          cur_scope, Outputs(kOutputs), scope, Outputs(kOutputs),
          [&](const framework::LoDTensor &src_tensor,
              framework::LoDTensor *dst_tensor) {
            if (i == 0) {  // create output tensor at begin
              dst_tensor->Resize(AppendSeqLenToDim(seq_len, src_tensor.dims()));
              dst_tensor->mutable_data(dev_ctx.GetPlace(), src_tensor.type());
            }

            auto dst_out = dst_tensor->Slice(seq_offset, seq_offset + 1);
            // Explicit copy output since the local RNN scope can be destroyed
            // early.
            dst_out.CopyFrom(src_tensor, dev_ctx.GetPlace(), dev_ctx);
          });

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
    auto *block = Attr<framework::BlockDescBind *>(kStepBlock);
    auto *program = block->Program();

    for (size_t step_id = 0; step_id < seq_len; ++step_id) {
      size_t seq_offset = reverse ? step_id : seq_len - step_id - 1;
      VLOG(3) << "Recurrent backward operate at the time step " << seq_offset;
      auto &cur_scope = scopes.CurScope();
      // Link outside::output_grads --> inside::output_grads
      //   inside::output_grad = outside::output_grad[seq_offset:seq_offset+1]
      AccessTensor(
          scope, Inputs(kOutputGrads), &cur_scope, Inputs(kOutputGrads),
          [&seq_offset](const framework::Tensor &outside,
                        framework::Tensor *inside) {
            inside->ShareDataWith(outside.Slice(seq_offset, seq_offset + 1));
            auto dims = framework::vectorize(inside->dims());
            dims.erase(dims.begin());
            inside->Resize(framework::make_ddim(dims));
          });

      // Link memories
      //   ex_scope::ex_state_grad --> cur_scope::cur_state_grad
      if (step_id != 0) {  // not at beginning
        auto &ex_scope = scopes.ExScope();
        ShareVars(
            ex_scope, GradVarLists(Attr<std::vector<std::string>>(kExStates)),
            &cur_scope, GradVarLists(Attr<std::vector<std::string>>(kStates)));
      }

      // Run
      executor.Run(*program, &cur_scope, block->ID(),
                   false /*create_local_scope*/);

      auto local_var_names = LocalVarNames(cur_scope);

      // Accumulate params
      //   outside::param_grad += inside::param_grad
      {
        auto &pg_names = Outputs(kParamGrads);
        auto &p_names = Inputs(kParameters);
        PADDLE_ENFORCE_EQ(pg_names.size(), p_names.size());

        for (size_t prog_id = 0; prog_id < pg_names.size(); ++prog_id) {
          auto inside_grad_name = framework::GradVarName(p_names[prog_id]);

          // If does not compute gradient of that variable inside rnn, just
          // continue
          if (local_var_names.find(inside_grad_name) == local_var_names.end()) {
            continue;
          }

          // zero gradient variable in step 0
          if (step_id == 0) {
            auto &inside_tensor = cur_scope.FindVar(inside_grad_name)
                                      ->Get<framework::LoDTensor>();
            framework::AttributeMap attrs;

            attrs["data_type"] = framework::ToDataType(inside_tensor.type());
            attrs["shape"] = framework::vectorize2int(inside_tensor.dims());
            attrs["value"] = 0.0f;

            auto zero_op = framework::OpRegistry::CreateOp(
                "fill_constant", {}, {{"Out", {pg_names[prog_id]}}}, attrs);
            zero_op->Run(scope, dev_ctx);
          }

          // sum gradient
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
              "sum", {{"X", {result_var_name, inside_grad_name}}},
              {{"Out", {result_var_name}}}, {});
          sum_op->Run(cur_scope, dev_ctx);
        }
      }

      // Copy input gradient from inside to outside
      //   outside::input_grad[seq_offset: seq_offset + 1] = inside::input_grad
      AccessTensor(
          cur_scope, GradVarLists(Inputs(kInputs)), scope, Outputs(kInputGrads),
          [&](const framework::LoDTensor &inside,
              framework::LoDTensor *outside) {
            if (step_id == 0) {  // alloc memory
              outside->Resize(AppendSeqLenToDim(seq_len, inside.dims()));
              outside->mutable_data(dev_ctx.GetPlace(), inside.type());
            }

            auto dst = outside->Slice(seq_offset, seq_offset + 1);
            dst.CopyFrom(inside, dev_ctx.GetPlace(), dev_ctx);
          });

      if (step_id + 1 == seq_len) {  // at_end
        // copy initialize states gradient from inside to outside
        AccessTensor(cur_scope,
                     GradVarLists(Attr<std::vector<std::string>>(kExStates)),
                     scope, Outputs(kInitStateGrads),
                     [&](const framework::LoDTensor &inside,
                         framework::LoDTensor *outside) {
                       outside->Resize(inside.dims());
                       outside->mutable_data(dev_ctx.GetPlace(), inside.type());
                       outside->CopyFrom(inside, dev_ctx.GetPlace(), dev_ctx);
                     });
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
  static std::vector<std::string> GradVarLists(
      const std::vector<std::string> &var_names) {
    std::vector<std::string> retv;
    retv.reserve(var_names.size());
    std::transform(var_names.begin(), var_names.end(), std::back_inserter(retv),
                   framework::GradVarName);
    return retv;
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
    AddAttr<framework::BlockDescBind *>(kStepBlock, "");
    AddAttr<bool>(kReverse, "").SetDefault(false);
    AddAttr<bool>(kIsTrain, "").SetDefault(true);
    AddComment("");
  }
};

class RecurrentGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDescBind> Apply() const {
    auto *grad = new framework::OpDescBind();
    grad->SetType("recurrent_grad");
    for (auto &input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param));
    }

    for (auto &output_param : this->OutputNames()) {
      if (output_param == kStepScopes) {
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

    return std::unique_ptr<framework::OpDescBind>(grad);
  }
};

class RecurrentGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    std::vector<std::string> input{kInputs, kInitialStates};
    std::vector<std::string> output{kOutputs};
    for (auto &s : input) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
      PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(s)));
    }
    for (auto &s : output) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
    }

    for (auto &s : input) {
      ctx->SetOutputsDim(framework::GradVarName(s), ctx->GetInputsDim(s));
    }
    if (ctx->HasInputs(kParameters)) {
      PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(kParameters)));
      ctx->SetOutputsDim(framework::GradVarName(kParameters),
                         ctx->GetInputsDim(kParameters));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(recurrent, paddle::operators::RecurrentOp,
                  paddle::operators::RecurrentOpProtoMaker,
                  paddle::operators::RecurrentGradOpDescMaker);
REGISTER_OPERATOR(recurrent_grad, paddle::operators::RecurrentGradOp,
                  paddle::operators::RecurrentGradOpShapeInference);
