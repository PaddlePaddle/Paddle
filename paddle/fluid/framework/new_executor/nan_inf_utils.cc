// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/nan_inf_utils.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/selected_rows.h"

namespace paddle {
namespace framework {

static std::once_flag pir_white_list_init_flag;

void CheckTensorHasNanOrInf(InstructionBase* instruction,
                            const paddle::framework::Scope* scope,
                            ValueExecutionInfo* value_exe_info) {
  std::call_once(pir_white_list_init_flag, details::InitWhiteListFormEnv);

  std::string dialect_name = instruction->Operation()
                                 ->attributes()
                                 .at("op_name")
                                 .dyn_cast<pir::StrAttribute>()
                                 .AsString();
  auto api_name =
      dialect_name.substr(dialect_name.find(".") + 1, dialect_name.size());
  auto op_name = phi::TransToFluidOpName(api_name);

  if (details::op_type_nan_inf_white_list().count(api_name) != 0) {
    return;
  }

  for (auto iter : instruction->Outputs()) {
    auto tensor_name = value_exe_info->GetVarName(iter.first);
    bool need_check = true;
    if (details::op_var_nan_inf_white_list().count(api_name) != 0) {
      for (auto& white_vname :
           details::op_var_nan_inf_white_list().at(api_name)) {
        if (tensor_name.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
    }
    if (!need_check) continue;

    if (scope) {
      const phi::DenseTensor* dense_tensor{nullptr};
      Variable* var = scope->FindVar(tensor_name);
      if (!var) {
        VLOG(10) << "No var found for tensor_name: " << tensor_name;
        continue;
      }
      if (var->IsType<phi::DenseTensor>()) {
        dense_tensor = var->GetMutable<phi::DenseTensor>();
      } else if (var->IsType<phi::SelectedRows>()) {
        dense_tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
      } else {
        VLOG(10) << "Only DenseTensor,SelectedRows,DistTensor need to check, "
                 << tensor_name << " is no need.";
        break;
      }

      auto& place = dense_tensor->place();
      if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        paddle::framework::details::tensor_check<phi::GPUContext>(
            api_name, tensor_name, *dense_tensor, place);
#else
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "Tensor[%s] use gpu place. PaddlePaddle must compile with GPU.",
            tensor_name));
#endif
        continue;
      }
      paddle::framework::details::tensor_check<phi::CPUContext>(
          api_name, tensor_name, *dense_tensor, place);
    }
  }
}

}  // namespace framework
}  // namespace paddle
