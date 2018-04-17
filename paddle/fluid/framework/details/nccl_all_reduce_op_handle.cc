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

struct ReduceLoDTensor {
  const std::vector<LoDTensor> &src_tensors_;
  LoDTensor &dst_tensor_;

  ReduceLoDTensor(const std::vector<LoDTensor> &src, LoDTensor *dst)
      : src_tensors_(src), dst_tensor_(*dst) {}

  template <typename T>
  void operator()() const {
    PADDLE_ENFORCE(!src_tensors_.empty());
    auto &t0 = src_tensors_[0];
    PADDLE_ENFORCE_NE(t0.numel(), 0);
    dst_tensor_.Resize(t0.dims());
    T *dst = dst_tensor_.mutable_data<T>(platform::CPUPlace());
    std::copy(t0.data<T>(), t0.data<T>() + t0.numel(), dst);

    for (size_t i = 1; i < src_tensors_.size(); ++i) {
      auto &t = src_tensors_[i];
      PADDLE_ENFORCE_EQ(t.dims(), t0.dims());
      PADDLE_ENFORCE_EQ(t.type(), t0.type());
      std::transform(t.data<T>(), t.data<T>() + t.numel(), dst, dst,
                     [](T a, T b) -> T { return a + b; });
    }
  }
};

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

    std::vector<LoDTensor> lod_tensors;

    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto *s = local_scopes_[i];

      auto &lod_tensor = s->FindVar(var_name)->Get<LoDTensor>();
      lod_tensors.emplace_back(lod_tensor);
    }

    if (platform::is_gpu_place(lod_tensors[0].place())) {
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &p = places_[i];
        auto &lod_tensor = lod_tensors[i];
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
      auto &trg =
          *this->local_scopes_[0]->Var()->GetMutable<framework::LoDTensor>();

      // Reduce All Tensor to trg in CPU
      ReduceLoDTensor func(lod_tensors, &trg);
      VisitDataType(ToDataType(lod_tensors[0].type()), func);

      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &scope = local_scopes_[i];
        auto &p = places_[i];
        auto *var = scope->FindVar(var_name);
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
