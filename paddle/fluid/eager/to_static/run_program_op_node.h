// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <iostream>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"

#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace details {
using Tensor = paddle::experimental::Tensor;

static std::vector<std::string> GetTensorsName(const std::vector<Tensor> &ins) {
  std::vector<std::string> in_names;
  for (auto &in_t : ins) {
    in_names.emplace_back(in_t.name());
  }
  return in_names;
}

static std::vector<std::string> GetTensorsName(
    const std::vector<Tensor *> &ins) {
  std::vector<std::string> in_names;
  for (auto *in_t : ins) {
    in_names.emplace_back(in_t->name());
  }
  return in_names;
}

static void CheckInputVarStatus(const Tensor &tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.defined() && phi::DenseTensor::classof(tensor.impl().get()), true,
      paddle::platform::errors::InvalidArgument(
          "The input tensor %s of "
          "RunProgram(Grad)Op holds "
          "wrong type. Expect type is DenseTensor.",
          tensor.name()));

  PADDLE_ENFORCE_EQ(tensor.initialized(), true,
                    paddle::platform::errors::InvalidArgument(
                        "The tensor in input tensor %s of "
                        "RunProgram(Grad)Op "
                        "is not initialized.",
                        tensor.name()));
}

static void CheckOutputVarStatus(const paddle::framework::Variable &src_var,
                                 const Tensor &dst_tensor) {
  auto name = dst_tensor.name();
  PADDLE_ENFORCE_EQ(dst_tensor.defined(), true,
                    paddle::platform::errors::InvalidArgument(
                        "dst_tensor shall be defined."));

  if (phi::DenseTensor::classof(dst_tensor.impl().get())) {
    auto &src_tensor = src_var.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::DenseTensor::classof(&src_tensor), true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is DenseTensor",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.initialized(), true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          name));
  } else if (phi::SelectedRows::classof(dst_tensor.impl().get())) {
    auto &src_tensor = src_var.Get<phi::SelectedRows>();
    PADDLE_ENFORCE_EQ(phi::SelectedRows::classof(&src_tensor), true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensodfr %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is SelectedRows",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.initialized(), true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          name));

  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type LoDTensor or SelectedRows",
        name));
  }
}

static void ShareTensorsIntoScope(const std::vector<Tensor> &tensors,
                                  paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto name = tensors[i].name();
    if (name == "Fake_var") {
      continue;
    }
    auto *var = scope->Var(name);
    CheckInputVarStatus(tensors[i]);
    // share tensor
    auto tensor_base = tensors[i].impl();
    if (phi::DenseTensor::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor_base);
      *dst_tensor = *t;
    } else if (phi::SelectedRows::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::SelectedRows>();
      auto t = std::dynamic_pointer_cast<phi::SelectedRows>(tensor_base);
      *dst_tensor = *t;
    }
  }
}

static void ShareTensorsFromScope(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't find them in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    auto &name = tensors[i]->name();
    if (name == paddle::framework::kEmptyVarName || name == "Fake_var" ||
        !global_block.HasVar(name)) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(var, paddle::platform::errors::NotFound(
                                     "The output tensor %s is not in "
                                     "RunProgram(Grad)Op'"
                                     "s internal scope.",
                                     name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    auto tensor_base = tensors[i]->impl();
    if (phi::DenseTensor::classof(tensor_base.get())) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    } else if (phi::SelectedRows::classof(tensor_base.get())) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    }
  }
}

}  // namespace details

inline void RunProgramAPI(
    const std::vector<paddle::experimental::Tensor> &x,
    const std::vector<paddle::experimental::Tensor> &params,
    std::vector<paddle::experimental::Tensor *> &out,     // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    std::vector<paddle::experimental::Tensor *> &dout,    // NOLINT
    const paddle::framework::AttributeMap &attrs) {
  VLOG(2) << "RunProgramOpKernel Compute";
  auto start_op_index = BOOST_GET_CONST(int64_t, attrs.at("start_op_index"));
  auto end_op_index = BOOST_GET_CONST(int64_t, attrs.at("end_op_index"));
  auto is_test = BOOST_GET_CONST(bool, attrs.at("is_test"));
  auto program_id = BOOST_GET_CONST(int64_t, attrs.at("program_id"));

  // NOTE(chenweihang): In order not to add new variable type, use vector
  // here. Originally, here can use scope directly.
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(), 1,
      paddle::platform::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  // Step 2. prepare executor and init persistable variables

  // NOTE(Aurelius84): While training some models, forward can be called many
  // times and then apply backpropagation all at once, such as Reinforcement
  // Learning. Tensor data in multi-step training should be saved into single
  // scope separately. Otherwise, the gradients can be miscalculated because
  // always using the Tensor data of the last step in forward.
  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();
  VLOG(2) << "The number of sub scopes before forward: "
          << out_scope_vec->front()->kids().size();
  paddle::framework::Scope &scope = global_inner_scope->NewScope();

  // share input_vars & parameters into scope
  details::ShareTensorsIntoScope(x, &scope);
  details::ShareTensorsIntoScope(params, &scope);

  auto *global_block =
      BOOST_GET_CONST(paddle::framework::BlockDesc *, attrs.at("global_block"));
  const auto &place = egr::Controller::Instance().GetExpectedPlace();

  if (end_op_index > start_op_index) {
    auto input_names = details::GetTensorsName(x);
    auto output_names = details::GetTensorsName(out);
    auto dout_names = details::GetTensorsName(dout);
    auto *program = global_block->Program();

    auto cache_info = paddle::framework::GetExecutorInfoFromCache(
        *program, place, start_op_index, end_op_index,
        /*is_grad=*/false, program_id, &scope);
    auto &parallel_executor = cache_info.first;
    // all out_vars are skip_eager_var
    auto &skip_eager_delete_vars =
        paddle::framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
            program_id, false);
    if (cache_info.second /*is_new_created*/) {
      parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_names);
      skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                    output_names.begin(), output_names.end());
      skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                    dout_names.begin(), dout_names.end());
      paddle::framework::details::ParseSafeEagerDeletionSkipVars(
          *program, end_op_index, output_names, &skip_eager_delete_vars);
    }

    // Step 3. run ops
    parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
  }
  // Step 4. Get Output
  details::ShareTensorsFromScope(out, *global_block, &scope);
  details::ShareTensorsFromScope(dout, *global_block, &scope);

  // Debug info: scope info when run end
  VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());
  // Step 5. Drop all children scopes while testing.
  if (is_test) {
    out_scope_vec->front()->DropKids();
  }
  VLOG(2) << "The number of sub scopes after forward: "
          << out_scope_vec->front()->kids().size();
  // #ifdef PADDLE_WITH_MKLDNN
  //     if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
  // #endif
}

class GradNodeRunProgram : public egr::GradNodeBase {
 public:
  GradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}

  ~GradNodeRunProgram() override = default;
  // Functor: perform backward computations
  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>> &grads)
      override {
    std::cout << "grads.size() : " << grads.size() << std::endl;
    if (grads.size() > 0) {
      std::cout << "grads[0].size() : " << grads[0].size() << std::endl;
    }
    paddle::experimental::Tensor out;
    // TODO(dev): Add RunProgramGradAPI

    return {{out}};
  }

  void SetTensorWrappers_X(
      const std::vector<paddle::experimental::Tensor> &tensors) {
    std::cout << "GradNodeRunProgram::SetTensorWrappers_X" << std::endl;
  }
};
