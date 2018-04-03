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

#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"

namespace paddle {
namespace framework {
namespace details {
NCCLAllReduceOpHandle::NCCLAllReduceOpHandle(
    const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    const platform::NCCLContextMap &ctxs)
    : local_scopes_(local_scopes), places_(places), nccl_ctxs_(ctxs) {
  for (auto &p : places_) {
    this->dev_ctxes_[p] = nccl_ctxs_.DevCtx(p);
  }
}

void NCCLAllReduceOpHandle::RunImpl() {
  if (inputs_.size() == 1) {
    return;  // No need to all reduce when GPU count = 1;
  } else {
    // Wait input done
    for (auto *in : inputs_) {
      auto &p = static_cast<VarHandle *>(in)->place_;
      in->generated_op_->Wait(dev_ctxes_[p]);
    }

    auto &var_name = static_cast<VarHandle *>(this->inputs_[0])->name_;
    int dtype = -1;
    size_t numel = 0;

    std::vector<std::function<void()>> all_reduce_calls;

    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &p = places_[i];
      auto *s = local_scopes_[i];
      int dev_id = boost::get<platform::CUDAPlace>(p).device;

      auto &lod_tensor = s->FindVar(var_name)->Get<LoDTensor>();
      void *buffer = const_cast<void *>(lod_tensor.data<void>());

      if (dtype == -1) {
        dtype = platform::ToNCCLDataType(lod_tensor.type());
      }

      if (numel == 0) {
        numel = static_cast<size_t>(lod_tensor.numel());
      }

      auto &nccl_ctx = nccl_ctxs_.at(dev_id);
      auto stream = nccl_ctx.stream();
      auto comm = nccl_ctx.comm_;
      all_reduce_calls.emplace_back([=] {
        PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
            buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
            comm, stream));
      });
    }

    platform::NCCLGroupGuard guard;
    for (auto &call : all_reduce_calls) {
      call();
    }
  }
}

std::string NCCLAllReduceOpHandle::Name() const { return "NCCL AllReduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
