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
#include "paddle/fluid/framework/details/gather_op_handle.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {
namespace details {

std::vector<VarHandle *> GetValidVarHandle(
    const std::vector<VarHandleBase *> &inputs) {
  std::vector<VarHandle *> in_var_handles;
  for (auto *in : inputs) {
    auto *in_handle = dynamic_cast<VarHandle *>(in);
    if (in_handle) {
      in_var_handles.push_back(in_handle);
    }
  }
  return in_var_handles;
}

void ReduceOpHandle::RunImpl() {
  // the input and output may have dummy var.
  std::vector<VarHandle *> in_var_handles = GetValidVarHandle(inputs_);
  std::vector<VarHandle *> out_var_handles = GetValidVarHandle(outputs_);

  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");
  PADDLE_ENFORCE_EQ(out_var_handles.size(), 1,
                    "The number of output should be one.");

  // Wait input done, this Wait is asynchronous operation
  if (in_var_handles[0]->generated_op_) {
    for (auto *in : in_var_handles) {
      auto &in_p = in->place_;
      in_var_handles[0]->generated_op_->Wait(dev_ctxes_[in_p]);
    }
  }

  // check in the same place
  auto in_0_handle = static_cast<VarHandle *>(in_var_handles[0]);
  auto pre_place = in_0_handle->place_;

  std::vector<platform::Place> in_places;
  for (auto *in_handle : in_var_handles) {
    auto in_p = in_handle->place_;
    PADDLE_ENFORCE_EQ(in_p.which(), pre_place.which(),
                      "Places must be all on CPU or all on CUDA.");
    in_places.emplace_back(in_p);
  }

  auto out_var = local_scopes_[out_var_handles[0]->scope_idx_]->FindVar(
      out_var_handles[0]->name_);

  auto pre_in_var =
      local_scopes_[in_0_handle->scope_idx_]->FindVar(in_0_handle->name_);

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    auto &pre_in = pre_in_var->Get<framework::SelectedRows>();
    std::vector<const SelectedRows *> in_selected_rows;

    for (auto *in_handle : in_var_handles) {
      auto in_var =
          local_scopes_.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
      auto &in_sr = in_var->Get<framework::SelectedRows>();

      PADDLE_ENFORCE_EQ(in_sr.value().type(), pre_in.value().type(),
                        "The type of input is not consistent.");

      in_selected_rows.emplace_back(&in_sr);
    }
    auto trg = out_var->GetMutable<framework::SelectedRows>();
    GatherSelectedRows(in_selected_rows, in_places, dev_ctxes_,
                       out_var_handles[0]->place_, trg);
  } else {
    auto pre_in = pre_in_var->Get<framework::LoDTensor>();
    std::vector<LoDTensor> lod_tensors;

    // can be refined
    for (auto *in_handle : in_var_handles) {
      auto in_var =
          local_scopes_.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
      auto &in_sr = in_var->Get<framework::LoDTensor>();

      PADDLE_ENFORCE_EQ(in_sr.type(), pre_in.type(),
                        "The type of input is not consistent.");

      lod_tensors.emplace_back(in_sr);
    }

    auto trg = out_var->GetMutable<framework::LoDTensor>();
    trg->Resize(pre_in.dims());
    trg->mutable_data(out_var_handles[0]->place_, pre_in.type());

    if (paddle::platform::is_cpu_place(pre_place)) {
      ReduceLoDTensor func(lod_tensors, trg);
      VisitDataType(ToDataType(lod_tensors[0].type()), func);

    } else if (paddle::platform::is_gpu_place(pre_place)) {
#ifdef PADDLE_WITH_CUDA
      auto out_p = out_var_handles[0]->place_;
      int root = boost::get<platform::CUDAPlace>(out_p).device;

      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = lod_tensors[i];
        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_.at(dev_id);
        auto stream = nccl_ctx.stream();
        auto comm = nccl_ctx.comm_;

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root == dev_id) {
          recvbuffer = trg->mutable_data(out_var_handles[0]->place_);
        }

        all_reduce_calls.emplace_back([=] {
          PADDLE_ENFORCE(platform::dynload::ncclReduce(
              buffer, recvbuffer, static_cast<size_t>(lod_tensor.numel()),
              platform::ToNCCLDataType(lod_tensor.type()), ncclSum, root, comm,
              stream));
        });
      }

      platform::NCCLGroupGuard guard;
      for (auto &call : all_reduce_calls) {
        call();
      }
#else
      PADDLE_THROW("CUDA is not support.");
#endif
    } else {
      PADDLE_THROW("Error");
    }
  }
}
std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
