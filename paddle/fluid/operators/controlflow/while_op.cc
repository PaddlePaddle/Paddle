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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;

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

static void TransferVariablePlace(const framework::Scope *scope,
                                  const std::string &var_name,
                                  const phi::Place &dst_place,
                                  const platform::DeviceContext &dev_ctx) {
  framework::Variable *var = scope->FindVar(var_name);
  if (var == nullptr) {
    VLOG(4) << "[TransferVariablePlace]"
            << "lost in_var: " << var_name;
    return;
  }
  if (var->Type() != framework::proto::VarType::LOD_TENSOR) {
    VLOG(10) << "[TransferVariablePlace]" << var_name << " type changed:"
             << framework::TransToPhiDataType(
                    framework::ToVarType(var->Type()));
    return;
  }
  phi::DenseTensor *t = var->GetMutable<phi::DenseTensor>();
  if (t->place() == dst_place) {
    VLOG(10) << "[TransferVariablePlace]"
             << "no need transfer: " << var_name;
    return;
  }

  phi::DenseTensor *new_t = new phi::DenseTensor;
  framework::TensorCopy(*t, dst_place, new_t);
  dev_ctx.Wait();

  t->set_meta(new_t->meta());
  t->ResetHolder(new_t->Holder());

  VLOG(4) << "[TransferVariablePlace]" << var_name
          << " place: " << new_t->place();
}

}  // namespace

class WhileOp : public framework::OperatorBase {
 public:
  WhileOp(const std::string &type,
          const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Input(kCondition)),
                            platform::errors::NotFound(
                                "Input(Condition) of WhileOp is not found."));

    auto &cond = scope.FindVar(Input(kCondition))->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        cond.dims(),
        phi::make_ddim({1}),
        platform::errors::InvalidArgument(
            "The shape of Input(Condition) of WhileOp must be 1. But now "
            "the Condition's shape is ",
            cond.dims().to_str(),
            ".\n"));

#ifdef PADDLE_WITH_MKLDNN
    // (jczaja) Executor on being destroyed clears oneDNN cache and
    // resets registered model data layout. This is unwanted for nested
    // Executors (executors declared inside control ops)
    platform::DontClearMKLDNNCache(dev_place);
#endif
    auto *block = Attr<framework::BlockDesc *>(kStepBlock);

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto *program = block->Program();
    bool is_test = Attr<bool>("is_test");

    std::set<std::string> no_copy_var_names;
    if (!is_test) {
      // set all persistable parameters into no_copy_var_names.
      auto *global_block = block;

      while (global_block->ID() != 0)
        global_block = global_block->ParentBlock();
      auto all_vars = global_block->AllVars();
      std::for_each(all_vars.begin(),
                    all_vars.end(),
                    [&no_copy_var_names](framework::VarDesc *var) {
                      if (var->IsParameter())
                        no_copy_var_names.insert(var->Name());
                    });

      const std::vector<framework::OpDesc *> &all_ops = block->AllOps();
      for (const framework::OpDesc *op : all_ops) {
        const framework::VariableNameMap &input_var_names = op->Inputs();
        const framework::VariableNameMap &output_var_names = op->Outputs();
        for (auto &ipt : input_var_names) {
          for (const std::string &var_name : ipt.second) {
            if (StrInVaraiableNameMap(var_name, output_var_names)) {
              no_copy_var_names.insert(var_name);
            }
          }
        }
      }
    }

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

    PADDLE_ENFORCE_EQ(step_scopes->size(),
                      0,
                      platform::errors::PreconditionNotMet(
                          "The Output(StepScope) of WhileOp should be empty."));

    bool cond_data = GetCondData(cond);
    auto &skip_vars = Attr<std::vector<std::string>>(kSkipEagerDeletionVars);
    VLOG(2) << GetSkipEagerDeletionVarsDebugString(skip_vars);

    // note(lvyongkang): The assign op in while loop may change the place of
    // variable. However, InterpreterCore fix the kernel of every ops during its
    // first run. A cpu tensor may become gpu tensor after first run. This will
    // lead to segmetation fault when it's used in a cpu kernel. Here we record
    // the place of every inputs and restore their place after
    // InterpreterCore.run().
    std::map<std::string, phi::Place> input_var_original_places;
    for (const auto &in_name : Inputs(kX)) {
      framework::Variable *var = scope.FindVar(in_name);
      if (var == nullptr) {
        VLOG(4) << "[while op]"
                << "input not found:" << in_name;
      }

      if (var->Type() == framework::proto::VarType::LOD_TENSOR) {
        input_var_original_places[in_name] =
            (var->Get<phi::DenseTensor>()).place();
      } else {
        VLOG(10) << "[while op]"
                 << "skip backup input " << in_name << " type:"
                 << framework::TransToPhiDataType(
                        framework::ToVarType(var->Type()));
      }
    }

    if (FLAGS_control_flow_use_new_executor) {
      LOG_FIRST_N(INFO, 1) << "[ControlFlow][WhileOp] New Executor is Running.";
      if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
        std::set<std::string> skip_gc_vars(skip_vars.begin(), skip_vars.end());
        framework::Scope placeholder;  // Don't care if it's valid, just for
                                       // initialize InterpreterCore
        core_.reset(new framework::InterpreterCore(
            dev_place,
            *block,
            skip_gc_vars,
            &placeholder,
            /* used_for_jit */ false,
            /* used_for_control_flow_op */ true));
      }
    } else {
      if (!executor_ ||
          !platform::is_same_place(executor_->GetPlace(), dev_place)) {
        executor_.reset(new framework::Executor(dev_place));
        ctx_ = executor_->Prepare(*program, block->ID(), skip_vars);
      }
    }

    if (!is_test) {
      while (cond_data) {
        auto &current_scope = scope.NewScope();
        step_scopes->push_back(&current_scope);

        std::vector<std::string> rename_vars;
        for (const std::string &input_var_name : Inputs(kX)) {
          if (no_copy_var_names.find(input_var_name) ==
              no_copy_var_names.end()) {
            std::string input_var_rename = input_var_name + kSuffix;
            framework::Variable *input_var = scope.FindVar(input_var_name);
            if (input_var->IsType<phi::DenseTensor>()) {
              rename_vars.push_back(input_var_rename);
              auto input_var_tensor = input_var->Get<phi::DenseTensor>();
              auto *rename_input_var_tensor =
                  current_scope.Var(input_var_rename)
                      ->GetMutable<phi::DenseTensor>();
              framework::TensorCopy(
                  input_var_tensor, dev_place, rename_input_var_tensor);
              rename_input_var_tensor->set_lod(input_var_tensor.lod());
            }
          }
        }
        if (FLAGS_control_flow_use_new_executor) {
          BuildScopeForControlFlowOp(*core_, *block, &current_scope);
          core_->reset_scope(&current_scope);
          core_->Run({}, false);

          // restore inputs place
          for (const auto &n : input_var_original_places) {
            const std::string &in_name = n.first;
            const phi::Place &original_place = n.second;
            // input vars exist in `scope` not `current_scope`
            TransferVariablePlace(&scope, in_name, original_place, dev_ctx);
          }

        } else {
          executor_->RunPreparedContext(
              ctx_.get(), &current_scope, false, true, true);
        }

        for (auto &var_rename : rename_vars) {
          std::string input_var_name =
              var_rename.substr(0, var_rename.size() - strlen(kSuffix));
          current_scope.Rename(var_rename, input_var_name);
        }
        cond_data = GetCondData(
            scope.FindVar(Input(kCondition))->Get<phi::DenseTensor>());
      }
    } else {
      auto &current_scope = scope.NewScope();

      if (FLAGS_control_flow_use_new_executor) {
        BuildScopeForControlFlowOp(*core_, *block, &current_scope);
        core_->reset_scope(&current_scope);
      } else {
        executor_->CreateVariables(*program, &current_scope, block->ID());
      }

      while (cond_data) {
        for (auto &name : current_scope.LocalVarNames()) {
          auto *var = current_scope.Var(name);
          if (var->IsType<phi::DenseTensor>()) {
            // Clear all lod information for all lod_tensors.
            auto *t = var->GetMutable<phi::DenseTensor>();
            framework::LoD empty_lod;
            t->set_lod(empty_lod);
          } else if (var->IsType<framework::LoDTensorArray>()) {
            // Clear elements of all tensor arrays.
            auto *t = var->GetMutable<framework::LoDTensorArray>();
            t->clear();
          }
        }

        if (FLAGS_control_flow_use_new_executor) {
          core_->Run({}, false);
        } else {
          executor_->RunPreparedContext(
              ctx_.get(), &current_scope, false, false, false);
        }

        cond_data = GetCondData(
            scope.FindVar(Input(kCondition))->Get<phi::DenseTensor>());
      }

      scope.DeleteScope(&current_scope);
    }
  }

 private:
  mutable std::shared_ptr<framework::Executor> executor_{nullptr};
  mutable std::unique_ptr<framework::ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};
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
    AddComment(R"DOC(
)DOC");
  }
};

class WhileGradOp : public framework::OperatorBase {
 public:
  WhileGradOp(const std::string &type,
              const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_ENFORCE_EQ(
        Attr<bool>("is_test"),
        false,
        platform::errors::InvalidArgument(
            "WhileGradOp is only callable when is_test is false."));
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto *block = Attr<framework::BlockDesc *>(kStepBlock);
    auto *program = block->Program();
    auto *parent_block = block->ParentBlock();

    auto &skip_vars = Attr<std::vector<std::string>>(kSkipEagerDeletionVars);
    VLOG(2) << GetSkipEagerDeletionVarsDebugString(skip_vars);

    auto *step_scopes =
        scope.FindVar(Input(kStepScopes))->GetMutable<StepScopeVar>();

    auto outside_og_names = Inputs(framework::GradVarName(kOutputs));
    auto inside_og_names =
        Attr<std::vector<std::string>>("original_output_grad");

    PADDLE_ENFORCE_EQ(outside_og_names.size(),
                      inside_og_names.size(),
                      platform::errors::InvalidArgument(
                          "The number of original output gradient names "
                          "does not match the number of backward input "
                          "gradient names. The number of Backward input "
                          "names is %d and the numbers of original output "
                          "gradient names is %d.",
                          outside_og_names.size(),
                          inside_og_names.size()));

    if (FLAGS_control_flow_use_new_executor) {
      LOG_FIRST_N(INFO, 1)
          << "[ControlFlow][WhileGradOp] New Executor is Running.";
      if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
        std::set<std::string> skip_gc_vars(skip_vars.begin(), skip_vars.end());
        framework::Scope placeholder;  // Don't care if it's valid, just for
                                       // initialize InterpreterCore
        core_.reset(new framework::InterpreterCore(
            dev_place,
            *block,
            skip_gc_vars,
            &placeholder,
            /* used_for_jit */ false,
            /* used_for_control_flow_op */ true));
      }
    } else {
      if (!executor_ ||
          !platform::is_same_place(executor_->GetPlace(), dev_place)) {
        executor_.reset(new framework::Executor(dev_place));
        ctx_ = executor_->Prepare(*program, block->ID(), skip_vars);
      }
    }

    for (auto cur_scope_iter = step_scopes->rbegin();
         cur_scope_iter != step_scopes->rend();
         ++cur_scope_iter) {
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

        if (cur_scope_iter == step_scopes->rbegin()) {
          auto &og_outside = *scope.FindVar(outside_og_name);
          if (og_outside.IsType<phi::DenseTensor>() &&
              !og_outside.GetMutable<phi::DenseTensor>()->IsInitialized()) {
            auto *var_desc = parent_block->FindVarRecursive(outside_og_name);
            PADDLE_ENFORCE_NOT_NULL(var_desc,
                                    platform::errors::PreconditionNotMet(
                                        "Var `%s` is not found in parent "
                                        "block, can't fill constant.",
                                        outside_og_name));
            auto shape = var_desc->GetShape();
            VLOG(8) << "Found uninitialized tensor " << outside_og_name
                    << " in step 0, fill it with 0.0f. dims="
                    << phi::make_ddim(shape);
            framework::AttributeMap attrs;
            attrs["dtype"] = var_desc->GetDataType();
            attrs["shape"] = phi::vectorize<int>(phi::make_ddim(shape));
            attrs["value"] = 0.0f;

            auto var_name = outside_og_name;
            auto zero_op =
                framework::OpRegistry::CreateOp("fill_constant",
                                                framework::VariableNameMap{},
                                                {{"Out", {var_name}}},
                                                attrs);
            zero_op->Run(scope, dev_place);
          }
        }

        auto &og_outside = *scope.FindVar(outside_og_name);
        auto &og_inside = *cur_scope.Var(inside_og_name);
        if (og_outside.IsType<phi::DenseTensor>()) {
          auto &outside_tensor = og_outside.Get<phi::DenseTensor>();
          auto &inside_tensor = *og_inside.GetMutable<phi::DenseTensor>();
          inside_tensor.set_lod(outside_tensor.lod());
          inside_tensor.ShareDataWith(outside_tensor);
        } else if (og_outside.IsType<framework::LoDTensorArray>()) {
          auto outside_array =
              og_outside.GetMutable<framework::LoDTensorArray>();
          auto &inside_array =
              *og_inside.GetMutable<framework::LoDTensorArray>();
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
              PADDLE_ENFORCE_EQ(
                  inside_array[j].numel(),
                  0,
                  platform::errors::InvalidArgument(
                      "The numel of %d-th element of var %s (LoDTensorArray) "
                      "in while block must be 0, but received its numel is %d.",
                      j,
                      inside_og_name,
                      inside_array[j].numel()));
            }
          }
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Currently only support phi::DenseTensor and "
              "phi::DenseTensorArray in "
              "WhileGradOp."));
        }
      }

      if (FLAGS_control_flow_use_new_executor) {
        BuildScopeForControlFlowOp(*core_, *block, *cur_scope_iter);
        core_->reset_scope(*cur_scope_iter);
        core_->Run({}, false);
      } else {
        executor_->RunPreparedContext(
            ctx_.get(), *cur_scope_iter, false, true, true);
      }

      // The Outputs(kXGRAD) contains the names of the gradient of parameters
      // and inputs.
      auto &pg_ig_names = Outputs(kXGRAD);
      auto &p_names = Inputs(kX);
      PADDLE_ENFORCE_EQ(pg_ig_names.size(),
                        p_names.size(),
                        platform::errors::PreconditionNotMet(
                            "The number of names in Outputs(X@GRAD) does not "
                            "match the number of names in Inputs(X). The "
                            "number of names in Outputs(X@GRAD) is %d and "
                            "the number of names in Inputs(X) is %d.",
                            pg_ig_names.size(),
                            p_names.size()));
      for (size_t param_id = 0; param_id < pg_ig_names.size(); ++param_id) {
        if (pg_ig_names[param_id] == framework::kEmptyVarName) {
          continue;  // parameter doesn't have gradient
        }
        auto inside_grad_name = framework::GradVarName(p_names[param_id]);

        // for some grad_op, their input doesn't have gradient,
        // for example lookup_table_grad_op, the input(Idx) doesn't have
        // gradient.
        auto pg_ig_var = cur_scope.FindVar(inside_grad_name);
        PADDLE_ENFORCE_NOT_NULL(
            pg_ig_var,
            platform::errors::NotFound("Variable %s is not found.",
                                       inside_grad_name));
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

        auto is_var_input_and_output =
            std::find(outside_og_names.begin(),
                      outside_og_names.end(),
                      pg_ig_names[param_id]) != outside_og_names.end();

        // zero gradient variable in step 0
        if (cur_scope_iter == step_scopes->rbegin()) {
          auto *var = (*cur_scope_iter)->FindVar(inside_grad_name);
          PADDLE_ENFORCE_NOT_NULL(
              var,
              platform::errors::NotFound("Variable %s is not found.",
                                         inside_grad_name));
          PADDLE_ENFORCE_EQ(
              var->IsType<framework::LoDTensorArray>() ||
                  var->IsType<phi::DenseTensor>(),
              true,
              platform::errors::InvalidArgument(
                  "Currently the type of var only can be LoDTensorArray, "
                  "or phi::DenseTensor, but the received var[%s] is %s.",
                  inside_grad_name,
                  framework::ToTypeName(var->Type())));

          if (!is_var_input_and_output && var->IsType<phi::DenseTensor>()) {
            auto &inside_tensor = var->Get<phi::DenseTensor>();
            framework::AttributeMap attrs;
            attrs["dtype"] =
                framework::TransToProtoVarType(inside_tensor.dtype());
            attrs["shape"] = phi::vectorize<int>(inside_tensor.dims());
            attrs["value"] = 0.0f;

            auto var_name = pg_ig_names[param_id];
            auto zero_op =
                framework::OpRegistry::CreateOp("fill_constant",
                                                framework::VariableNameMap{},
                                                {{"Out", {var_name}}},
                                                attrs);
            zero_op->Run(scope, dev_place);
            scope.FindVar(var_name)->GetMutable<phi::DenseTensor>()->set_lod(
                inside_tensor.lod());
          }
        }
        if (!is_var_input_and_output) {
          auto new_inside_name = cur_scope.Rename(inside_grad_name);
          auto sum_op = framework::OpRegistry::CreateOp(
              "sum",
              {{"X", {pg_ig_names[param_id], new_inside_name}}},
              {{"Out", {pg_ig_names[param_id]}}},
              framework::AttributeMap{{"use_mkldnn", {false}}});
          sum_op->Run(cur_scope, dev_place);
          cur_scope.Rename(new_inside_name, inside_grad_name);
        } else {
          ShareVariable(cur_scope, scope, pg_ig_names[param_id]);
        }
      }
      dev_ctx.Wait();
      const_cast<framework::Scope &>(scope).DeleteScope(&cur_scope);
    }
    step_scopes->clear();
  }

  void ShareVariable(const framework::Scope &source,
                     const framework::Scope &dest,
                     std::string name) const {
    auto from_var = source.FindVar(name);
    auto to_var = dest.FindVar(name);
    if (from_var->IsType<phi::DenseTensor>()) {
      if (from_var->Get<phi::DenseTensor>().IsInitialized()) {
        to_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
            from_var->Get<phi::DenseTensor>());
      }
    } else if (from_var->IsType<framework::LoDTensorArray>()) {
      auto from_arr = from_var->GetMutable<framework::LoDTensorArray>();
      auto to_arr = to_var->GetMutable<framework::LoDTensorArray>();
      to_arr->clear();
      to_arr->resize(from_arr->size());
      for (size_t i = 0; i < to_arr->size(); ++i) {
        if (from_arr->at(i).IsInitialized()) {
          to_arr->at(i).ShareDataWith(from_arr->at(i));
        }
      }
    }
  }

 private:
  mutable std::shared_ptr<framework::Executor> executor_{nullptr};
  mutable std::unique_ptr<framework::ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};
};

template <typename T>
class WhileGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> while_grad) const override {
    while_grad->SetType("while_grad");
    while_grad->SetInput(kX, this->Input(kX));
    while_grad->SetInput(kOutputs, this->Output(kOutputs));
    while_grad->SetInput(kStepScopes, this->Output(kStepScopes));

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
    auto igs = this->InputGrad(kX, /*do not drop empty gradient*/ false);

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
    block_ins.reserve(this->Input(kX).size() + this->Output(kOutputs).size());
    for (auto &p : this->Input(kX)) {
      block_ins.insert(p);
    }
    for (auto &o : this->Output(kOutputs)) {
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
    std::copy(
        output_grads.begin(), output_grads.end(), output_grads_list.begin());
    while_grad->SetInput(framework::GradVarName(kOutputs), output_grads_list);

    while_grad->SetAttrMap(this->Attrs());
    while_grad->SetBlockAttr(kStepBlock, grad_block);
    // record the original output gradient names, since the gradient name of
    // while operator could be renamed.
    while_grad->SetAttr("original_output_grad", output_grads_list);

    while_grad->SetAttr(kSkipEagerDeletionVars, std::vector<std::string>());
  }
};

class WhileGradOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto p_names = Input(ctx, kX);
    auto pg_ig_names = Output(ctx, framework::GradVarName(kX));

    for (size_t i = 0; i < p_names.size(); ++i) {
      if (HasVar(ctx, pg_ig_names[i])) {
        VLOG(5) << "Setting " << pg_ig_names[i] << " following " << p_names[i]
                << " type: " << GetType(ctx, p_names[i]);
        SetType(ctx, pg_ig_names[i], GetType(ctx, p_names[i]));
        SetDataType(ctx, pg_ig_names[i], GetDataType(ctx, p_names[i]));
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
    auto in_var_ptrs = ctx->GetInputVarPtrs(kX);
    auto out_var_ptrs = ctx->GetOutputVarPtrs(kXGRAD);
    PADDLE_ENFORCE_EQ(in_var_ptrs.size(),
                      out_var_ptrs.size(),
                      platform::errors::InvalidArgument(
                          "The size of Inputs(X) must be the same as "
                          "the size of Outputs(X@GRAD)."));

    for (size_t i = 0; i < in_var_ptrs.size(); ++i) {
      if (pg_ig_names[i] == framework::kEmptyVarName) {
        continue;
      }
      framework::VarDesc *in_var =
          PADDLE_GET(framework::VarDesc *, in_var_ptrs[i]);
      PADDLE_GET(framework::VarDesc *, out_var_ptrs[i])
          ->SetShape(in_var->GetShape());
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    while,
    paddle::operators::WhileOp,
    paddle::operators::WhileOpMaker,
    paddle::operators::WhileGradOpMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(while_grad,
                  paddle::operators::WhileGradOp,
                  paddle::operators::WhileGradOpShapeInference,
                  paddle::operators::WhileGradOpVarTypeInference);
