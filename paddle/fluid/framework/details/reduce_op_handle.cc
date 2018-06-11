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

#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"

namespace paddle {
namespace framework {
namespace details {

void ReduceOpHandle::RunImpl() {
  if (places_.size() == 1) return;
  // the input and output may have dummy var.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);

  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  VarHandle *out_var_handle;
  {
    auto out_var_handles = DynamicCast<VarHandle>(outputs_);

    PADDLE_ENFORCE_EQ(out_var_handles.size(), 1,
                      "The number of output should be one.");
    out_var_handle = out_var_handles.front();
  }

  auto in_0_handle = in_var_handles[0];

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    var_scopes.emplace_back(s->FindVar(kLocalExecScopeName)->Get<Scope *>());
  }

  auto pre_in_var =
      var_scopes.at(in_0_handle->scope_idx_)->FindVar(in_0_handle->name_);
  PADDLE_ENFORCE_NOT_NULL(pre_in_var);

  // Wait input done, this Wait is asynchronous operation
  WaitInputVarGenerated();

  // NOTE: The Places of all input tensor must be all on CPU or all on GPU.
  std::vector<platform::Place> in_places;  // used to get dev_ctx
  for (auto *in_handle : in_var_handles) {
    in_places.emplace_back(in_handle->place_);
    auto in_var =
        var_scopes.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
    PADDLE_ENFORCE_NOT_NULL(in_var);
    VariableVisitor::EnforceShapeAndDTypeEQ(*pre_in_var, *in_var);
  }

  auto out_var =
      var_scopes.at(out_var_handle->scope_idx_)->FindVar(out_var_handle->name_);
  PADDLE_ENFORCE_NOT_NULL(out_var);

  // NOTE: The tensors' Place of input and output must be all on GPU or all on
  // CPU.
  auto in_p = VariableVisitor::GetMutableTensor(pre_in_var).place();
  platform::Place t_out_p;
  if (platform::is_gpu_place(in_p)) {
    PADDLE_ENFORCE(platform::is_gpu_place(out_var_handle->place_),
                   "Places of input and output must be all on GPU.");
    t_out_p = out_var_handle->place_;
  } else {
    t_out_p = platform::CPUPlace();
  }

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    this->RunAndRecordEvent([&] {
      std::vector<const SelectedRows *> in_selected_rows =
          GetInputValues<SelectedRows>(in_var_handles, var_scopes);
      GatherSelectedRows(in_selected_rows, in_places, dev_ctxes_, t_out_p,
                         out_var->GetMutable<framework::SelectedRows>());
    });
  } else {
    std::vector<const LoDTensor *> lod_tensors =
        GetInputValues<LoDTensor>(in_var_handles, var_scopes);
    if (paddle::platform::is_cpu_place(lod_tensors[0]->place())) {
      this->RunAndRecordEvent([&] {
        ReduceLoDTensor func(lod_tensors,
                             out_var->GetMutable<framework::LoDTensor>());
        VisitDataType(ToDataType(lod_tensors[0]->type()), func);
      });
    } else if (paddle::platform::is_gpu_place(lod_tensors[0]->place())) {
#ifdef PADDLE_WITH_CUDA
      auto pre_in = pre_in_var->Get<framework::LoDTensor>();
      VariableVisitor::ShareDimsAndLoD(*pre_in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(
          out_var_handle->place_, pre_in.type());

      auto out_p = out_var_handle->place_;
      int root_id = boost::get<platform::CUDAPlace>(out_p).device;
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < var_scopes.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = *lod_tensors[i];

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root_id == dev_id) {
          recvbuffer =
              out_var->GetMutable<framework::LoDTensor>()->mutable_data(
                  out_var_handle->place_);
        }

        int type = platform::ToNCCLDataType(lod_tensor.type());
        size_t numel = static_cast<size_t>(lod_tensor.numel());
        all_reduce_calls.emplace_back(
            [buffer, recvbuffer, type, numel, root_id, &nccl_ctx] {
              PADDLE_ENFORCE(platform::dynload::ncclReduce(
                  buffer, recvbuffer, numel, static_cast<ncclDataType_t>(type),
                  ncclSum, root_id, nccl_ctx.comm_, nccl_ctx.stream()));
            });
      }

      this->RunAndRecordEvent([&] {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      });
#else
      PADDLE_THROW("CUDA is not enabled.");
#endif
    } else {
      PADDLE_THROW("Place should be CPUPlace or CUDAPlace.");
    }
  }
}

template <typename T>
std::vector<const T *> ReduceOpHandle::GetInputValues(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<const Scope *> &var_scopes) const {
  std::vector<const T *> in_selected_rows;
  for (auto *in_handle : in_var_handles) {
    auto &in_sr = var_scopes.at(in_handle->scope_idx_)
                      ->FindVar(in_handle->name_)
                      ->Get<T>();
    in_selected_rows.emplace_back(&in_sr);
  }
  return in_selected_rows;
}

std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
