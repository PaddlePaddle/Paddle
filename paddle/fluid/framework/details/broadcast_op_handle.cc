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
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

void BroadcastOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  if (places_.size() == 1) return;

  // The input and output may have dummy vars.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  PADDLE_ENFORCE_EQ(in_var_handles.size(), 1UL,
                    "The number of input should be one.");
  PADDLE_ENFORCE_EQ(
      out_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  VarHandle *in_var_handle = in_var_handles[0];

  WaitInputVarGenerated();

  BroadcastOneVar(*in_var_handle, out_var_handles, local_exec_scopes_);
}

void BroadcastOpHandle::BroadcastOneVar(
    const VarHandle &in_var_handle,
    const std::vector<VarHandle *> &out_var_handles,
    const std::vector<Scope *> &var_scopes) {
  auto *in_var =
      var_scopes.at(in_var_handle.scope_idx())->FindVar(in_var_handle.name());
  PADDLE_ENFORCE_NOT_NULL(in_var);
  Tensor &in_tensor = VariableVisitor::GetMutableTensor(in_var);
  if (UNLIKELY(!in_tensor.IsInitialized())) {
    VLOG(3) << "in var " << in_var_handle.name() << "not inited, return!";
    return;
  }

  InitOutputValue(in_var_handle, out_var_handles);

  if (platform::is_cpu_place(in_tensor.place())) {
    for (auto *out_var_handle : out_var_handles) {
      if (out_var_handle->IsTheSameVar(in_var_handle)) {
        continue;
      }
      auto &out_p = out_var_handle->place();
      auto *out_var = var_scopes.at(out_var_handle->scope_idx())
                          ->FindVar(out_var_handle->name());

      RunAndRecordEvent(out_p, [in_tensor, out_var] {
        paddle::framework::TensorCopy(
            in_tensor, platform::CPUPlace(),
            &VariableVisitor::GetMutableTensor(out_var));
      });
    }
  } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    VarHandle *out_handle = nullptr;
    int root_id = boost::get<platform::CUDAPlace>(in_tensor.place()).device;
    std::vector<std::function<void()>> broadcast_calls;

    int type = platform::ToNCCLDataType(in_tensor.type());
    size_t numel = static_cast<size_t>(in_tensor.numel());

    for (auto out_var_handle : out_var_handles) {
      Variable *out_var = var_scopes.at(out_var_handle->scope_idx())
                              ->FindVar(out_var_handle->name());

      int dst_id =
          boost::get<platform::CUDAPlace>(out_var_handle->place()).device;

      auto &nccl_ctx = nccl_ctxs_->at(dst_id);

      void *send_recv_buffer = nullptr;
      if (root_id == dst_id) {
        send_recv_buffer = const_cast<void *>(in_tensor.data<void>());
        out_handle = out_var_handle;
      } else {
        send_recv_buffer = VariableVisitor::GetMutableTensor(out_var)
                               .Resize(in_tensor.dims())
                               .mutable_data(out_var_handle->place());
      }

      broadcast_calls.emplace_back(
          [send_recv_buffer, numel, type, root_id, &nccl_ctx] {
            PADDLE_ENFORCE(platform::dynload::ncclBcast(
                send_recv_buffer, numel, static_cast<ncclDataType_t>(type),
                root_id, nccl_ctx.comm_, nccl_ctx.stream()));
          });
    }

    this->RunAndRecordEvent([&] {
      {
        platform::NCCLGroupGuard guard;
        for (auto &call : broadcast_calls) {
          call();
        }
      }

      if (!out_handle->IsTheSameVar(in_var_handle)) {
        auto out_var = var_scopes.at(in_var_handle.scope_idx())
                           ->FindVar(out_var_handles[0]->name());
        paddle::framework::TensorCopy(
            in_tensor, in_var_handle.place(),
            *(dev_ctxes_.at(in_var_handle.place())),
            &VariableVisitor::GetMutableTensor(out_var));
      }
    });
#else
    PADDLE_THROW("CUDA is not enabled.");
#endif
  }
}

void BroadcastOpHandle::InitOutputValue(
    const VarHandle &in_var_handle,
    const std::vector<VarHandle *> &out_var_handles) const {
  auto &var_scopes = local_exec_scopes_;
  auto *in_var =
      var_scopes.at(in_var_handle.scope_idx())->FindVar(in_var_handle.name());

  Tensor &in_tensor = VariableVisitor::GetMutableTensor(in_var);

  // NOTE: The tensors' Place of input and output must be all on GPU or all on
  // CPU.
  for (auto *out_var_handle : out_var_handles) {
    if (out_var_handle->IsTheSameVar(in_var_handle)) {
      continue;
    }
    auto t_out_p = out_var_handle->place();
    auto *out_var = var_scopes.at(out_var_handle->scope_idx())
                        ->FindVar(out_var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(out_var);
    if (is_gpu_place(in_tensor.place())) {
      PADDLE_ENFORCE(platform::is_gpu_place(t_out_p),
                     "Places of input and output must be all on GPU.");
    } else {
      t_out_p = platform::CPUPlace();
    }
    VariableVisitor::ShareDimsAndLoD(*in_var, out_var);
    VariableVisitor::GetMutableTensor(out_var).mutable_data(t_out_p,
                                                            in_tensor.type());
  }
}

std::string BroadcastOpHandle::Name() const { return "broadcast"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
