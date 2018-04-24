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

#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/variable_visitor.h"

namespace paddle {
namespace framework {
namespace details {

void BroadcastOpHandle::RunImpl() {
  if (places_.size() == 1) return;
  // the input and output may have dummy var.
  VarHandle *in_var_handle;

  {
    auto in_var_handles = DynamicCast<VarHandle>(inputs_);
    PADDLE_ENFORCE_EQ(in_var_handles.size(), 1,
                      "The number of input should be one.");
    in_var_handle = in_var_handles[0];
  }

  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  PADDLE_ENFORCE_EQ(
      out_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  // Wait input done, this Wait is asynchronous operation platform::Place
  // &in_place;
  WaitInputVarGenerated(*in_var_handle);

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    var_scopes.emplace_back(s->FindVar(kLocalExecScopeName)->Get<Scope *>());
  }

  auto *in_var =
      var_scopes.at(in_var_handle->scope_idx_)->FindVar(in_var_handle->name_);
  PADDLE_ENFORCE_NOT_NULL(in_var);

  Tensor &in_tensor = VariableVisitor::GetMutableTensor(in_var);

  if (platform::is_cpu_place(in_tensor.place())) {
    for (auto *out : out_var_handles) {
      if (*out == *in_var_handle) {
        continue;
      }

      auto &out_p = out->place_;
      auto *out_var = var_scopes.at(out->scope_idx_)->FindVar(out->name_);
      PADDLE_ENFORCE_NOT_NULL(out_var);
      PADDLE_ENFORCE_EQ(out_p.which(), in_tensor.place().which(),
                        "Places must be all on CPU or all on CUDA.");

      VariableVisitor::ShareDimsAndLoD(*in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(out_p,
                                                              in_tensor.type());

      auto dev_ctx = dev_ctxes_.at(out_p);
      RunAndRecordEvent(out_p, [in_tensor, out_var, dev_ctx, out_p] {
        paddle::framework::TensorCopy(
            in_tensor, out_p, *dev_ctx,
            &VariableVisitor::GetMutableTensor(out_var));
      });
    }
  } else {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(platform::is_gpu_place(in_tensor.place()));
    VarHandle *out_handle;
    int root = boost::get<platform::CUDAPlace>(in_tensor.place()).device;
    std::vector<std::function<void()>> broadcast_calls;

    for (size_t j = 0; j < out_var_handles.size(); ++j) {
      VarHandle *out_var_handle = out_var_handles[j];
      Variable *out_var = var_scopes.at(out_var_handle->scope_idx_)
                              ->FindVar(out_var_handle->name_);

      if (*out_var_handle != *in_var_handle) {
        PADDLE_ENFORCE_NOT_NULL(out_var);
        PADDLE_ENFORCE_EQ(out_var_handle->place_.which(),
                          in_tensor.place().which(),
                          "Places must be all on CPU or all on CUDA.");
        VariableVisitor::ShareDimsAndLoD(*in_var, out_var);
        VariableVisitor::GetMutableTensor(out_var).mutable_data(
            out_var_handle->place_, in_tensor.type());
      }

      auto out_p = out_var_handle->place_;
      int dev_id = boost::get<platform::CUDAPlace>(out_p).device;

      auto &nccl_ctx = nccl_ctxs_->at(dev_id);
      auto stream = nccl_ctx.stream();
      auto comm = nccl_ctx.comm_;

      void *send_recv_buffer = nullptr;
      if (root == dev_id) {
        send_recv_buffer = const_cast<void *>(in_tensor.data<void>());
        out_handle = out_var_handle;
      } else {
        send_recv_buffer =
            VariableVisitor::GetMutableTensor(out_var).mutable_data(
                out_var_handle->place_);
      }

      int type = platform::ToNCCLDataType(in_tensor.type());
      broadcast_calls.emplace_back([=] {
        PADDLE_ENFORCE(platform::dynload::ncclBcast(
            send_recv_buffer, in_tensor.numel(),
            static_cast<ncclDataType_t>(type), root, comm, stream));
      });
    }

    this->RunAndRecordEvent([&] {
      {
        platform::NCCLGroupGuard guard;
        for (auto &call : broadcast_calls) {
          call();
        }
      }
      if (*out_handle != *in_var_handle) {
        auto out_var = var_scopes.at(in_var_handle->scope_idx_)
                           ->FindVar(out_var_handles[0]->name_);
        paddle::framework::TensorCopy(
            in_tensor, in_var_handle->place_,
            *(dev_ctxes_.at(in_var_handle->place_)),
            &VariableVisitor::GetMutableTensor(out_var));
      }
    });
#else
    PADDLE_THROW("CUDA is not support.");
#endif
  }
}

void BroadcastOpHandle::WaitInputVarGenerated(const VarHandle &in_var) {
  if (in_var.generated_op_) {
    for (auto &pair : dev_ctxes_) {
      in_var.generated_op_->Wait(pair.second);
    }
  }
}

std::string BroadcastOpHandle::Name() const { return "broadcast"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
