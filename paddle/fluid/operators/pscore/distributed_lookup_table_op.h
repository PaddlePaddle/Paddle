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
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DistributedLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &scope = context.scope();

    auto padding_idx = context.Attr<int64_t>("padding_idx");
    auto table_id = context.Attr<int>("table_id");
    bool is_test = context.Attr<bool>("is_test");

    auto embedding_name = context.InputNames("W").front();
    int64_t emb_dim = 0;

    auto *var = scope.FindVar(embedding_name);

    if (var->IsType<framework::LoDTensor>()) {
      emb_dim = var->Get<framework::LoDTensor>().dims()[1];
    } else if (var->IsType<pten::SelectedRows>()) {
      emb_dim = var->Get<pten::SelectedRows>().value().dims()[1];
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of `W` must be Tensor, SelectedRows.But got "
          "unsupport type: %s.",
          framework::ToTypeName(var->Type())));
    }

    auto inputs = context.MultiInput<framework::LoDTensor>("Ids");
    auto outputs = context.MultiOutput<framework::LoDTensor>("Outputs");

    // auto fleet = distributed::FleetWrapper::GetInstance();
    auto *communicator = (distributed::AsyncCommunicator *)
        distributed::Communicator::GetInstance();

    if (platform::is_cpu_place(context.GetPlace())) {
      communicator->PullSparseToTensorSync(
          static_cast<uint64_t>(table_id), emb_dim,
          static_cast<uint64_t>(padding_idx), context.GetPlace(), !is_test,
          &inputs, &outputs);
    } else {
      auto inputs_variable = context.MultiInputVar("Ids");
      auto outputs_variable = context.MultiOutputVar("Outputs");
      auto inputs_name = context.InputNames("Ids");
      auto outputs_name = context.OutputNames("Outputs");

      auto cpu_place = platform::CPUPlace();
      framework::Scope *tmp_scope = scope.NewTmpScope().release();

      std::vector<const framework::LoDTensor *> tmp_input_vec;
      auto input_var_size = inputs_variable.size();
      std::vector<framework::LoDTensor *> tmp_output_vec;
      auto output_var_size = outputs_variable.size();

      // create temp input
      for (size_t idx = 0; idx < input_var_size; ++idx) {
        framework::Variable *tmp_input_var = tmp_scope->Var(inputs_name[idx]);
        framework::LoDTensor *tmp_input_tensor =
            tmp_input_var->GetMutable<framework::LoDTensor>();
        framework::TensorCopy(inputs_variable[idx]->Get<framework::LoDTensor>(),
                              cpu_place, context.device_context(),
                              tmp_input_tensor);
        tmp_input_vec.push_back(tmp_input_tensor);
      }

      // create temp output
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        framework::Variable *tmp_output_var = tmp_scope->Var(outputs_name[idx]);
        framework::LoDTensor *tmp_output_tensor =
            tmp_output_var->GetMutable<framework::LoDTensor>();
        tmp_output_tensor->Resize(outputs[idx]->dims());
        tmp_output_vec.push_back(tmp_output_tensor);
      }

      // use fleet->PullSparse
      communicator->PullSparseToTensorSync(
          static_cast<uint64_t>(table_id), emb_dim,
          static_cast<uint64_t>(padding_idx), cpu_place, !is_test,
          &tmp_input_vec, &tmp_output_vec);

      // cp temp to origin
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        framework::Variable *tmp_output_var = tmp_scope->Var(outputs_name[idx]);
        framework::LoDTensor *tmp_output_tensor =
            tmp_output_var->GetMutable<framework::LoDTensor>();
        framework::TensorCopy(
            *tmp_output_tensor, context.GetPlace(), context.device_context(),
            outputs_variable[idx]->GetMutable<framework::LoDTensor>());
      }
      delete tmp_scope;
    }

    auto id_names = context.InputNames("Ids");
    auto out_names = context.OutputNames("Outputs");
    auto lookup_table_version =
        context.Attr<std::string>("lookup_table_version");

    if (lookup_table_version == "lookup_table_v2") {
      for (size_t i = 0; i < id_names.size(); ++i) {
        auto *id_var = scope.FindVar(id_names[i]);
        auto *out_var = scope.FindVar(out_names[i]);
        auto *id_tensor = id_var->GetMutable<framework::LoDTensor>();
        auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

        auto id_dims = id_tensor->dims();
        out_tensor->Resize(framework::make_ddim(
            {static_cast<int64_t>(id_dims[0]), static_cast<int64_t>(id_dims[1]),
             static_cast<int64_t>(emb_dim)}));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
