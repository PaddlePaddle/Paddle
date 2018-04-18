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
  auto pre_place = in_0_handle->place_;

  // Wait input done, this Wait is asynchronous operation
  WaitInputVarGenerated(in_var_handles);

  std::vector<platform::Place> in_places;
  auto pre_in_tensor = VariableVisitor::GetMutableTensor(pre_in_var);
  for (auto *in_handle : in_var_handles) {
    auto in_p = in_handle->place_;
    PADDLE_ENFORCE_EQ(in_p.which(), pre_place.which(),
                      "Places must be all on CPU or all on CUDA.");
    in_places.emplace_back(in_p);

    auto in_var =
        var_scopes.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
    auto in_tensor = VariableVisitor::GetMutableTensor(in_var);

    PADDLE_ENFORCE_EQ(in_tensor.type(), pre_in_tensor.type(),
                      "The type of input is not consistent.");
  }

  auto out_var =
      var_scopes.at(out_var_handle->scope_idx_)->FindVar(out_var_handle->name_);

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    std::vector<const SelectedRows *> in_selected_rows;
    for (auto *in_handle : in_var_handles) {
      auto &in_sr = var_scopes.at(in_handle->scope_idx_)
                        ->FindVar(in_handle->name_)
                        ->Get<framework::SelectedRows>();
      in_selected_rows.emplace_back(&in_sr);
    }
    auto trg = out_var->GetMutable<framework::SelectedRows>();
    GatherSelectedRows(in_selected_rows, in_places, dev_ctxes_,
                       out_var_handle->place_, trg);
  } else {
    auto pre_in = pre_in_var->Get<framework::LoDTensor>();
    std::vector<LoDTensor> lod_tensors;
    for (auto *in_handle : in_var_handles) {
      lod_tensors.emplace_back(var_scopes.at(in_handle->scope_idx_)
                                   ->FindVar(in_handle->name_)
                                   ->Get<framework::LoDTensor>());
    }

    auto trg = out_var->GetMutable<framework::LoDTensor>();
    trg->set_lod(pre_in.lod());
    trg->Resize(pre_in.dims());
    trg->mutable_data(out_var_handle->place_, pre_in.type());

    if (paddle::platform::is_cpu_place(pre_place)) {
      ReduceLoDTensor func(lod_tensors, trg);
      VisitDataType(ToDataType(lod_tensors[0].type()), func);
    } else if (paddle::platform::is_gpu_place(pre_place)) {
#ifdef PADDLE_WITH_CUDA
      auto out_p = out_var_handle->place_;
      int root = boost::get<platform::CUDAPlace>(out_p).device;

      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < var_scopes.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = lod_tensors[i];

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);
        auto stream = nccl_ctx.stream();
        auto comm = nccl_ctx.comm_;

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root == dev_id) {
          recvbuffer = trg->mutable_data(out_var_handle->place_);
        }

        all_reduce_calls.emplace_back([=] {
          PADDLE_ENFORCE(platform::dynload::ncclReduce(
              buffer, recvbuffer, static_cast<size_t>(lod_tensor.numel()),
              platform::ToNCCLDataType(lod_tensor.type()), ncclSum, root, comm,
              stream));
        });
      }

      this->RunAndRecordEvent([&] {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      });
#else
      PADDLE_THROW("CUDA is not support.");
#endif
    } else {
      PADDLE_THROW("Place should be CPUPlace or CUDAPlace.");
    }
  }
}

void ReduceOpHandle::WaitInputVarGenerated(
    const std::vector<VarHandle *> &in_var_handles) {
  for (auto *in : in_var_handles) {
    if (in->generated_op_) {
      for (auto pair : dev_ctxes_) {
        in->generated_op_->Wait(pair.second);
      }
    }
  }
}

std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
