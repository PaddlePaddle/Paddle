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

#include "paddle/fluid/jit/function_utils.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::jit::utils {

std::vector<DenseTensor> ToDenseTensors(const std::vector<Tensor> &tensors) {
  std::vector<DenseTensor> ret;
  for (auto &t : tensors) {
    ret.emplace_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
  }
  return ret;
}

std::vector<Tensor> ToTensors(const std::vector<DenseTensor> &tensors) {
  std::vector<Tensor> ret;
  for (auto &t : tensors) {
    ret.emplace_back(std::make_shared<DenseTensor>(t));
  }
  return ret;
}

void FetchOuts(const std::vector<std::string> &names,
               const framework::Scope &scope,
               std::vector<DenseTensor> *outs) {
  outs->reserve(names.size());
  for (const auto &out_name : names) {
    VLOG(3) << "fetch out: " << out_name;
    auto *var = scope.FindVar(out_name);
    auto &src_tensor = var->Get<DenseTensor>();
    outs->emplace_back(src_tensor);
  }
}

void ShareIntoScope(const std::vector<std::string> &ordered_input_names,
                    const std::vector<DenseTensor> &tensors,
                    framework::Scope *scope) {
  VLOG(3) << "tensors size: " << tensors.size();
  PADDLE_ENFORCE_EQ(
      tensors.size(),
      ordered_input_names.size(),
      common::errors::InvalidArgument(
          "tensors.size() should be equal to ordered_input_names.size()."));
  for (size_t i = 0; i < tensors.size(); ++i) {
    VLOG(3) << "share into scope: " << ordered_input_names[i];
    auto *var = scope->Var(ordered_input_names[i]);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = tensors[i];
  }
}

void ShareParamsIntoScope(const std::vector<std::string> &param_names,
                          const std::shared_ptr<VariableMap> &params_dict,
                          framework::Scope *scope) {
  for (auto name : param_names) {
    PADDLE_ENFORCE_EQ(params_dict->count(name),
                      1,
                      common::errors::InvalidArgument(
                          "Parameter named %s is not existed in params_dict. "
                          "Please check that your model was saved correctly",
                          name));

    auto &param = params_dict->find(name)->second;
    auto &dense_tensor = param->Get<DenseTensor>();
    auto *var = scope->Var(name);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void RemoveFeedFetch(framework::ProgramDesc *program_desc) {
  for (size_t i = 0; i < program_desc->Size(); ++i) {
    auto *block = program_desc->MutableBlock(i);
    const auto &all_ops = block->AllOps();
    size_t op_size = all_ops.size();
    VLOG(3) << "op_size: " << op_size;
    for (int i = static_cast<int>(op_size - 1); i >= 0; i--) {
      auto op = all_ops[i];
      if (op->Type() == "feed") {
        VLOG(3) << "remove op type: " << op->Type() << ", index: " << i
                << ", var name: " << op->Input("X")[0];
        block->RemoveVar(op->Input("X")[0]);
        block->RemoveOp(i, i + 1);
      } else if (op->Type() == "fetch") {
        VLOG(3) << "remove op type: " << op->Type() << ", index: " << i
                << ", var name: " << op->Output("Out")[0];
        block->RemoveVar(op->Output("Out")[0]);
        block->RemoveOp(i, i + 1);
      }
    }
  }
}

}  // namespace paddle::jit::utils
