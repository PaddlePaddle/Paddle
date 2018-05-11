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
#include <algorithm>
#include "paddle/fluid/framework/details/reduce_and_gather.h"

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
    WaitInputVarGenerated();

    auto &var_name = static_cast<VarHandle *>(this->inputs_[0])->name_;
    int dtype = -1;
    size_t numel = 0;

    std::vector<const LoDTensor *> lod_tensors;

    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto *s = local_scopes_[i];
      auto &local_scope = *s->FindVar(kLocalExecScopeName)->Get<Scope *>();

      auto &lod_tensor = local_scope.FindVar(var_name)->Get<LoDTensor>();
      lod_tensors.emplace_back(&lod_tensor);
    }

    if (platform::is_gpu_place(lod_tensors[0]->place())) {
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &p = places_[i];
        auto &lod_tensor = *lod_tensors[i];
        void *buffer = const_cast<void *>(lod_tensor.data<void>());

        if (dtype == -1) {
          dtype = platform::ToNCCLDataType(lod_tensor.type());
        }

        if (numel == 0) {
          numel = static_cast<size_t>(lod_tensor.numel());
        }

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_.at(dev_id);
        auto stream = nccl_ctx.stream();
        auto comm = nccl_ctx.comm_;
        all_reduce_calls.emplace_back([=] {
          PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
              buffer, buffer, numel, static_cast<ncclDataType_t>(dtype),
              ncclSum, comm, stream));
        });
      }
      this->RunAndRecordEvent([&] {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      });
    } else {  // Special handle CPU only Operator's gradient. Like CRF
      auto &trg = *this->local_scopes_[0]
                       ->FindVar(kLocalExecScopeName)
                       ->Get<Scope *>()
                       ->Var()
                       ->GetMutable<framework::LoDTensor>();

      // Reduce All Tensor to trg in CPU
      ReduceLoDTensor func(lod_tensors, &trg);
      VisitDataType(ToDataType(lod_tensors[0]->type()), func);

      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &scope =
            *local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope *>();
        auto &p = places_[i];
        auto *var = scope.FindVar(var_name);
        auto *dev_ctx = dev_ctxes_[p];

        RunAndRecordEvent(p, [&trg, var, dev_ctx, p] {
          auto &tensor_gpu = *var->GetMutable<framework::LoDTensor>();
          auto &tensor_cpu = trg;
          TensorCopy(tensor_cpu, p, *dev_ctx, &tensor_gpu);
        });
      }
    }
  }
}

std::string NCCLAllReduceOpHandle::Name() const { return "nccl_all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
