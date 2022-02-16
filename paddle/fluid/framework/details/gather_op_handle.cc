//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/gather_op_handle.h"

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/variable_visitor.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {
namespace details {

GatherOpHandle::GatherOpHandle(ir::Node *node,
                               const std::vector<Scope *> &local_scopes,
                               const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}

void GatherOpHandle::RunImpl() {
  if (places_.size() == 1) return;
  // the input and output may have dummy var.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);

  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      platform::errors::InvalidArgument(
          "The number of input variables should be equal "
          "to the number of places, but got the number of input variables is "
          "%d and the number of places is %d.",
          in_var_handles.size(), places_.size()));

  VarHandle *out_var_handle;
  {
    auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
    PADDLE_ENFORCE_EQ(
        out_var_handles.size(), 1,
        platform::errors::InvalidArgument(
            "The number of output variables should be 1, but got %d.",
            out_var_handles.size()));
    out_var_handle = out_var_handles.front();
  }

  auto &var_scopes = local_exec_scopes_;

  auto in_0_handle = in_var_handles[0];
  auto pre_in_var =
      var_scopes.at(in_0_handle->scope_idx())->FindVar(in_0_handle->name());
  PADDLE_ENFORCE_NOT_NULL(
      pre_in_var,
      platform::errors::NotFound("The variable '%s' is not found in the scope.",
                                 in_0_handle->name()));

  PADDLE_ENFORCE_EQ(pre_in_var->IsType<pten::SelectedRows>(), true,
                    platform::errors::Unimplemented(
                        "Currently, gather_op only supports SelectedRows."));

  // Wait input done, this Wait is asynchronous operation
  WaitInputVarGenerated();

  auto &pre_in_value = pre_in_var->Get<pten::SelectedRows>();
  std::vector<int64_t> out_rows;
  std::vector<Tensor> in_tensors;

  // Gather the inputs
  for (auto *in_handle : in_var_handles) {
    auto *in_var =
        var_scopes.at(in_handle->scope_idx())->FindVar(in_handle->name());
    PADDLE_ENFORCE_NOT_NULL(
        in_var,
        platform::errors::NotFound(
            "The variable '%s' is not found in the scope.", in_handle->name()));
    VariableVisitor::EnforceShapeAndDTypeEQ(*in_var, *pre_in_var);

    auto &in_sr_value = in_var->Get<pten::SelectedRows>();

    auto &in_sr_rows = in_sr_value.rows();
    out_rows.insert(out_rows.end(), in_sr_rows.begin(), in_sr_rows.end());
    in_tensors.emplace_back(in_sr_value.value());
  }

  // NOTE: The Places of all input tensor must be all on CPU or all on GPU.
  platform::Place t_out_p = out_var_handle->place();
  if (platform::is_gpu_place(pre_in_value.place())) {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(t_out_p), true,
                      platform::errors::PreconditionNotMet(
                          "Places of input and output must be all on GPU."));
  } else {
    t_out_p = platform::CPUPlace();
  }

  auto out_var = var_scopes.at(out_var_handle->scope_idx())
                     ->FindVar(out_var_handle->name());
  PADDLE_ENFORCE_NOT_NULL(
      out_var,
      platform::errors::NotFound("The variable '%s' is not found in the scope.",
                                 out_var_handle->name()));
  auto out_value = out_var->GetMutable<pten::SelectedRows>();
  out_value->set_height(pre_in_value.height());
  out_value->set_rows(out_rows);
  size_t rows = out_rows.size();
  DDim out_dim = pre_in_value.GetCompleteDims();
  out_dim[0] = static_cast<int64_t>(rows);
  out_value->mutable_value()->Resize(out_dim).mutable_data(
      t_out_p, pre_in_value.value().dtype());
  Tensor *out_tensor = out_value->mutable_value();

  // copy
  auto dev_ctx = dev_ctxes_.at(out_var_handle->place());
  RunAndRecordEvent(out_var_handle->place(), [in_tensors, out_tensor, &dev_ctx,
                                              t_out_p] {
    int s = 0, e = 0;
    for (size_t j = 0; j < in_tensors.size(); ++j) {
      e += in_tensors[j].dims()[0];
      auto sub_out = out_tensor->Slice(s, e);
      paddle::framework::TensorCopy(in_tensors[j], t_out_p, *dev_ctx, &sub_out);
      s = e;
    }
  });
}

std::string GatherOpHandle::Name() const { return "gather"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
