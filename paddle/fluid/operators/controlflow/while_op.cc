// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using LoDTensor = framework::LoDTensor;

namespace {  // NOLINT
static std::string GetSkipEagerDeletionVarsDebugString(
    const std::vector<std::string> &vars) {
  std::string str = "Skip " + std::to_string(vars.size()) +
                    " var(s) in eager deletion mode: ";
  for (auto &var : vars) {
    str.append(var);
    str.push_back(' ');
  }
  return str;
}
}  // NOLINT

class WhileOp : public framework::OperatorBase {
 public:
  WhileOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Input(kCondition)));

    auto &cond = scope.FindVar(Input(kCondition))->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(cond.dims(), paddle::framework::make_ddim({1}));

    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>(kStepBlock);

    auto *program = block->Program();

    auto step_scopes =
        scope.FindVar(Output(kStepScopes))->GetMutable<StepScopeVar>();

    if (step_scopes->size() > 0) {
      platform::DeviceContextPool::Instance().Get(dev_place)->Wait();
      for (auto &s : *step_scopes) {
        if (scope.HasKid(s)) {
          scope.DeleteScope(s);
        }
      }
      step_scopes->clear();
    }

    PADDLE_ENFORCE_EQ(step_scopes->size(), 0, "The StepScope should be empty.");
    PADDLE_ENFORCE(platform::is_cpu_place(cond.place()),
                   "Condition of while op must in CPU memory.");

    bool is_test = Attr<bool>("is_test");
    auto &skip_vars = Attr<std::vector<std::string>>(kSkipEagerDeletionVars);
    VLOG(2) << GetSkipEagerDeletionVarsDebugString(skip_vars);

    auto ctx = executor.Prepare(*program, block->ID(), skip_vars);
    if (!is_test) {
      while (cond.data<bool>()[0]) {
        auto &current_scope = scope.NewScope();
        step_scopes->push_back(&current_scope);
        executor.RunPreparedContext(ctx.get(), &current_scope, false, true,
                                    true);
      }
    } else {
      auto &current_scope = scope.NewScope();
      executor.CreateVariables(*program, &current_scope, block->ID());
      while (cond.data<bool>()[0]) {
        for (auto &name : current_scope.LocalVarNames()) {
          auto *var = current_scope.Var(name);
          if (var->IsType<framework::LoDTensor>()) {
            // Clear all lod information for all lod_tensors.
            auto *t = var->GetMutable<framework::LoDTensor>();
            framework::LoD empty_lod;
            t->set_lod(empty_lod);
          } else if (var->IsType<framework::LoDTensorArray>()) {
            // Clear elements of all tensor arrays.
            auto *t = var->GetMutable<framework::LoDTensorArray>();
            t->clear();
          }
        }
        executor.RunPreparedContext(ctx.get(), &current_scope, false, false,
                                    false);
      }
      scope.DeleteScope(&current_scope);
    }
  }
};

class WhileOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "A set of variables, which are required by operators inside the "
             "block of While Op.")
        .AsDuplicable();
    AddInput(
        kCondition,
        "(Bool) An scalar. When it's False, the While Op will be terminated.")
        .AsDuplicable();
    AddOutput(kOutputs,
              "A set of variables, which will be assigned with values "
              "generated by the operators inside the block of While Op.")
        .AsDuplicable();
    AddOutput(kStepScopes,
              "(StepScopeVar) A vector of local scope, which size equals the "
              "step number of While Op. The i'th scope storages temporary "
              "variables generated in the i'th step.");
    AddAttr<framework::BlockDesc *>(kStepBlock,
                                    "The step block inside WhileOp");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>(kSkipEagerDeletionVars,
                                      "Vars that would skip eager deletion."
                                      "Users should not set this manually.")
        .SetDefault(std::vector<std::string>());
    AddComment(R"DOC(
)DOC");
  }
};

class WhileGradOp : public framework::OperatorBase {
 public:
  WhileGradOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_ENFORCE(!Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>(kStepBlock);
    auto *program = block->Program();

    auto &skip_vars = Attr<std::vector<std::string>>(kSkipEagerDeletionVars);
    VLOG(2) << GetSkipEagerDeletionVarsDebugString(skip_vars);
    auto ctx = executor.Prepare(*program, block->ID(), skip_vars);

    auto *step_scopes =
        scope.FindVar(Input(kStepScopes))->GetMutable<StepScopeVar>();

    auto outside_og_names = Inputs(framework::GradVarName(kOutputs));
    auto inside_og_names =
        Attr<std::vector<std::string>>("original_output_grad");

    PADDLE_ENFORCE_EQ(outside_og_names.size(), inside_og_names.size());

    for (auto cur_scope_iter = step_scopes->rbegin();
         cur_scope_iter != step_scopes->rend(); ++cur_scope_iter) {
      VLOG(3) << "Start backward at time_step "
              << cur_scope_iter - step_scopes->rbegin();
      framework::Scope &cur_scope = **cur_scope_iter;
      // Link OG from outside to inside
      for (size_t i = 0; i < outside_og_names.size(); ++i) {
        auto outside_og_name = outside_og_names[i];
        auto inside_og_name = inside_og_names[i];
        VLOG(8) << "Linking outside " << outside_og_name << " --> inside "
                << inside_og_name;
        if (scope.FindVar(outside_og_name) == nullptr) {
          continue;
        }

        auto &og_outside =
            detail::Ref(scope.FindVar(outside_og_name),
                        "Cannot find Outside Gradient %s", outside_og_name);
        auto &og_inside =
            detail::Ref(cur_scope.Var(inside_og_name),
                        "Cannot find inside gradient %s", inside_og_name);
        if (og_outside.IsType<framework::LoDTensor>()) {
          auto &outside_tensor = og_outside.Get<framework::LoDTensor>();
          auto &inside_tensor =
              detail::Ref(og_inside.GetMutable<framework::LoDTensor>());
          inside_tensor.set_lod(outside_tensor.lod());
          inside_tensor.ShareDataWith(outside_tensor);
        } else if (og_outside.IsType<framework::LoDTensorArray>()) {
          auto outside_array =
              og_outside.GetMutable<framework::LoDTensorArray>();
          auto &inside_array =
              detail::Ref(og_inside.GetMutable<framework::LoDTensorArray>());
          inside_array.clear();
          inside_array.resize(outside_array->size());
          VLOG(8) << outside_og_name << " size = " << outside_array->size();

          for (size_t j = 0; j < inside_array.size(); ++j) {
            if (!outside_array->at(j).IsInitialized()) {
              outside_array->at(j).Resize({0});
            }
            VLOG(8) << j << " " << outside_array->at(j).numel();
            if (outside_array->at(j).numel() != 0) {
              inside_array[j].set_lod(outside_array->at(j).lod());
              inside_array[j].ShareDataWith(outside_array->at(j));
            } else {
              PADDLE_ENFORCE_EQ(inside_array[j].numel(), 0);
            }
          }
        } else {
          PADDLE_THROW("Currently only support LoDTensor and LoDTensorArray.");
        }
      }
      executor.RunPreparedContext(ctx.get(), *cur_scope_iter, false, true,
                                  true);

      // The Outputs(kXGRAD) contains the names of the gradient of parameters
      // and inputs.
      auto &pg_ig_names = Outputs(kXGRAD);
      auto &p_names = Inputs(kX);
      PADDLE_ENFORCE_EQ(pg_ig_names.size(), p_names.size());
      for (size_t param_id = 0; param_id < pg_ig_names.size(); ++param_id) {
        if (pg_ig_names[param_id] == framework::kEmptyVarName) {
          continue;  // parameter doesn't have gradient
        }
        auto inside_grad_name = framework::GradVarName(p_names[param_id]);

        // for some grad_op, their input doesn't have gradient,
        // for example lookup_table_grad_op, the input(Idx) doesn't have
        // gradient.
        auto pg_ig_var = cur_scope.FindVar(inside_grad_name);
        PADDLE_ENFORCE(pg_ig_var != nullptr);
        if (pg_ig_var->IsType<framework::LoDTensorArray>()) {
          auto pg_ig_lod_t_arr =
              pg_ig_var->GetMutable<framework::LoDTensorArray>();
          bool empty = true;
          for (auto &each : *pg_ig_lod_t_arr) {
            if (each.numel() != 0) {
              empty = false;
              break;
            }
          }
          if (empty) {
            LOG(WARNING) << pg_ig_names[param_id]
                         << " is not found in cur_scope.";
            continue;
          }
        }

        //  // TODO(tonyyang-svail): Not sure we need the following
        //  // If does not compute gradient of that variable inside rnn,
        //  just
        //  // continue
        //  if (local_var_names.find(inside_grad_name) ==
        //  local_var_names.end()) {
        //    continue;
        //  }

        // zero gradient variable in step 0
        if (cur_scope_iter == step_scopes->rbegin()) {
          auto *var = (*cur_scope_iter)->FindVar(inside_grad_name);
          PADDLE_ENFORCE_NOT_NULL(var, "Can not find var %s", inside_grad_name);
          PADDLE_ENFORCE(
              var->IsType<framework::LoDTensorArray>() ||
                  var->IsType<LoDTensor>(),
              "Currently the type of var only can be LoDTensorArray, "
              "or LoDTensor, but the received var[%s] is %s.",
              inside_grad_name, framework::ToTypeName(var->Type()));

          if (var->IsType<LoDTensor>()) {
            auto &inside_tensor = var->Get<framework::LoDTensor>();
            framework::AttributeMap attrs;
            attrs["dtype"] = inside_tensor.type();
            attrs["shape"] = framework::vectorize<int>(inside_tensor.dims());
            attrs["value"] = 0.0f;

            auto var_name = pg_ig_names[param_id];
            auto zero_op = framework::OpRegistry::CreateOp(
                "fill_constant", framework::VariableNameMap{},
                {{"Out", {var_name}}}, attrs);
            zero_op->Run(scope, dev_place);
            scope.FindVar(var_name)
                ->GetMutable<framework::LoDTensor>()
                ->set_lod(inside_tensor.lod());
          }
        }
        auto new_inside_name = cur_scope.Rename(inside_grad_name);
        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {pg_ig_names[param_id], new_inside_name}}},
            {{"Out", {pg_ig_names[param_id]}}},
            framework::AttributeMap{{"use_mkldnn", {false}}});
        sum_op->Run(cur_scope, dev_place);
        cur_scope.Rename(new_inside_name, inside_grad_name);
      }
      dev_ctx.Wait();
      const_cast<framework::Scope &>(scope).DeleteScope(&cur_scope);
    }
    step_scopes->clear();
  }
};

class WhileGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *while_grad = new framework::OpDesc();
    while_grad->SetType("while_grad");
    while_grad->SetInput(kX, Input(kX));
    while_grad->SetInput(kOutputs, Output(kOutputs));
    while_grad->SetInput(kStepScopes, Output(kStepScopes));

    auto *grad_block = this->grad_block_[0];
    auto *fwd_block = grad_block->ForwardBlock();
    auto *parent_block = grad_block->ParentBlock();

    // Not all of IGs will be generated by inner gradient operators of while op.
    // Ignore IGs that is not generated by the inside block.
    std::unordered_set<std::string> inner_op_outputs;
    for (const auto *op : grad_block->AllOps()) {
      for (auto &oname : op->OutputArgumentNames()) {
        inner_op_outputs.insert(oname);
      }
    }
    auto igs = InputGrad(kX, /*do not drop empty gradient*/ false);
    for (auto &each_ig : igs) {
      if (inner_op_outputs.find(each_ig) == inner_op_outputs.end()) {
        VLOG(8) << "Ignore " << each_ig;
        each_ig = framework::kEmptyVarName;
      }
    }
    while_grad->SetOutput(framework::GradVarName(kX), igs);

    // OG should be re-calculated by step blocks, since many outputs of while op
    // do not need to calculate gradients.
    std::unordered_set<std::string> block_ins;
    block_ins.reserve(Input(kX).size() + Output(kOutputs).size());
    for (auto &p : Input(kX)) {
      block_ins.insert(p);
    }
    for (auto &o : Output(kOutputs)) {
      block_ins.insert(o);
    }
    std::unordered_set<std::string> output_grads;
    for (const auto *op : grad_block->AllOps()) {
      for (auto &input_name : op->InputArgumentNames()) {
        // If the input of Op has been recorded or is generated by the forward
        // block, do not make it as input again.

        // The input is located in I/O or other op's outputs or the variable is
        // located in grad_block's parents
        if (block_ins.find(input_name) != block_ins.end() ||
            (fwd_block->FindVarRecursive(input_name) != nullptr ||
             parent_block->FindVarRecursive(input_name) != nullptr)) {
          continue;
        }

        output_grads.insert(input_name);
      }
      for (auto &output_name : op->OutputArgumentNames()) {
        block_ins.insert(output_name);
      }
    }

    std::vector<std::string> output_grads_list;
    output_grads_list.resize(output_grads.size());
    std::copy(output_grads.begin(), output_grads.end(),
              output_grads_list.begin());
    while_grad->SetInput(framework::GradVarName(kOutputs), output_grads_list);

    while_grad->SetAttrMap(this->Attrs());
    while_grad->SetBlockAttr(kStepBlock, grad_block);
    // record the original output gradient names, since the gradient name of
    // while operator could be renamed.
    while_grad->SetAttr("original_output_grad", output_grads_list);

    while_grad->SetAttr(kSkipEagerDeletionVars, std::vector<std::string>());

    return std::unique_ptr<framework::OpDesc>(while_grad);
  }
};

class WhileGradOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto p_names = ctx->Input(kX);
    auto pg_ig_names = ctx->Output(framework::GradVarName(kX));

    for (size_t i = 0; i < p_names.size(); ++i) {
      if (ctx->HasVar(pg_ig_names[i])) {
        VLOG(5) << "Setting " << pg_ig_names[i] << " following " << p_names[i]
                << " type: " << ctx->GetType(p_names[i]);
        ctx->SetType(pg_ig_names[i], ctx->GetType(p_names[i]));
        ctx->SetDataType(pg_ig_names[i], ctx->GetDataType(p_names[i]));
      }
    }
  }
};

class WhileGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    ctx->HasInputs(kX);
    ctx->HasOutputs(framework::GradVarName(kX));
    ctx->HasInputs(kOutputs);
    ctx->HasInputs(framework::GradVarName(kOutputs));

    auto pg_ig_names = ctx->Outputs(kXGRAD);
    std::vector<framework::InferShapeVarPtr> in_var_ptrs =
        ctx->GetInputVarPtrs(kX);
    std::vector<framework::InferShapeVarPtr> out_var_ptrs =
        ctx->GetOutputVarPtrs(kXGRAD);
    PADDLE_ENFORCE(in_var_ptrs.size() == out_var_ptrs.size());

    for (size_t i = 0; i < in_var_ptrs.size(); ++i) {
      if (pg_ig_names[i] == framework::kEmptyVarName) {
        continue;
      }
      if (ctx->IsRuntime()) {
        framework::Variable *in_var =
            boost::get<framework::Variable *>(in_var_ptrs[i]);
        framework::Variable *out_var =
            boost::get<framework::Variable *>(out_var_ptrs[i]);

        auto type = framework::ToVarType(in_var->Type());
        if (type == framework::proto::VarType::LOD_TENSOR) {
          out_var->GetMutable<LoDTensor>()->Resize(
              in_var->Get<framework::LoDTensor>().dims());
        } else if (type == framework::proto::VarType::SELECTED_ROWS) {
          out_var->GetMutable<framework::SelectedRows>()->set_height(
              in_var->Get<framework::SelectedRows>().GetCompleteDims()[0]);
        } else if (type == framework::proto::VarType::LOD_TENSOR_ARRAY) {
          PADDLE_THROW("WhileGradOp doesn't support type %d",
                       static_cast<int>(type));
        }
      } else {
        framework::VarDesc *in_var =
            boost::get<framework::VarDesc *>(in_var_ptrs[i]);
        boost::get<framework::VarDesc *>(out_var_ptrs[i])
            ->SetShape(in_var->GetShape());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(while, paddle::operators::WhileOp,
                  paddle::operators::WhileOpMaker,
                  paddle::operators::WhileGradOpDescMaker);
REGISTER_OPERATOR(while_grad, paddle::operators::WhileGradOp,
                  paddle::operators::WhileGradOpShapeInference,
                  paddle::operators::WhileGradOpVarTypeInference);
