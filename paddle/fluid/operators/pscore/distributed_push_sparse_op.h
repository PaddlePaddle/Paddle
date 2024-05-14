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
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class DistributedPushSparseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &scope = context.scope();

    auto padding_idx = context.Attr<int64_t>("padding_idx");
    auto table_id = context.Attr<int>("table_id");
    auto emb_dim = context.Attr<int>("size");
    auto use_cvm_op = context.Attr<bool>("use_cvm_op");
    auto slots = context.Attr<std::vector<int>>("slots");

    auto inputs = context.MultiInput<phi::DenseTensor>("Ids");
    auto shows = context.Input<phi::DenseTensor>("Shows");
    auto clks = context.Input<phi::DenseTensor>("Clicks");
    auto outputs = context.MultiOutput<phi::DenseTensor>("Outputs");

    auto fleet = distributed::FleetWrapper::GetInstance();

    if (context.GetPlace().GetType() == phi::AllocationType::CPU) {
      fleet->PushSparseFromTensorAsync(static_cast<uint64_t>(table_id),
                                       emb_dim,
                                       static_cast<uint64_t>(padding_idx),
                                       context.GetPlace(),
                                       &inputs,
                                       slots,
                                       shows,
                                       clks,
                                       &outputs,
                                       use_cvm_op);
    } else {
      auto inputs_variable = context.MultiInputVar("Ids");
      auto outputs_variable = context.MultiOutputVar("Outputs");
      auto inputs_name = context.InputNames("Ids");
      auto outputs_name = context.OutputNames("Outputs");

      auto cpu_place = phi::CPUPlace();
      framework::Scope *tmp_scope = scope.NewTmpScope().release();

      std::vector<const phi::DenseTensor *> tmp_input_vec;
      auto input_var_size = inputs_variable.size();
      std::vector<phi::DenseTensor *> tmp_output_vec;
      auto output_var_size = outputs_variable.size();

      // create temp input
      for (size_t idx = 0; idx < input_var_size; ++idx) {
        framework::Variable *tmp_input_var = tmp_scope->Var(inputs_name[idx]);
        phi::DenseTensor *tmp_input_tensor =
            tmp_input_var->GetMutable<phi::DenseTensor>();
        framework::TensorCopy(inputs_variable[idx]->Get<phi::DenseTensor>(),
                              cpu_place,
                              context.device_context(),
                              tmp_input_tensor);
        tmp_input_vec.push_back(tmp_input_tensor);
      }

      framework::Variable *tmp_shows_var = tmp_scope->Var("Shows");
      phi::DenseTensor *tmp_shows_tensor =
          tmp_shows_var->GetMutable<phi::DenseTensor>();
      framework::Variable *tmp_clicks_var = tmp_scope->Var("Clicks");
      phi::DenseTensor *tmp_clicks_tensor =
          tmp_clicks_var->GetMutable<phi::DenseTensor>();
      framework::TensorCopy(
          *shows, cpu_place, context.device_context(), tmp_shows_tensor);
      framework::TensorCopy(
          *clks, cpu_place, context.device_context(), tmp_clicks_tensor);

      // create temp output
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        framework::Variable *tmp_output_var = tmp_scope->Var(outputs_name[idx]);
        phi::DenseTensor *tmp_output_tensor =
            tmp_output_var->GetMutable<phi::DenseTensor>();
        tmp_output_tensor->Resize(outputs[idx]->dims());
        tmp_output_vec.push_back(tmp_output_tensor);
      }

      // use fleet->PullSparse
      fleet->PushSparseFromTensorAsync(static_cast<uint64_t>(table_id),
                                       emb_dim,
                                       static_cast<uint64_t>(padding_idx),
                                       context.GetPlace(),
                                       &tmp_input_vec,
                                       slots,
                                       tmp_shows_tensor,
                                       tmp_clicks_tensor,
                                       &tmp_output_vec);

      // cp temp to origin
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        framework::Variable *tmp_output_var = tmp_scope->Var(outputs_name[idx]);
        phi::DenseTensor *tmp_output_tensor =
            tmp_output_var->GetMutable<phi::DenseTensor>();
        framework::TensorCopy(
            *tmp_output_tensor,
            context.GetPlace(),
            context.device_context(),
            outputs_variable[idx]->GetMutable<phi::DenseTensor>());
      }
      delete tmp_scope;
    }
  }
};

}  // namespace operators
}  // namespace paddle
