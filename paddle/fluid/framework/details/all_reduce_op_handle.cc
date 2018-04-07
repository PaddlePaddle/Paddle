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

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

AllReduceOpHandle::AllReduceOpHandle(const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLContextMap &ctxs,
                                     const int device_count)
    : local_scopes_(local_scopes),
      places_(places),
      nccl_ctxs_(ctxs),
      device_count_(device_count) {
  for (auto &p : places_) {
    this->dev_ctxes_[p] = nccl_ctxs_.DevCtx(p);
  }
}

void AllReduceOpHandle::RunImpl() {
  auto in_var0 = static_cast<VarHandle *>(this->inputs_[0]);
  auto &var_name = in_var0->name_;

  // Wait input done, this Wait is asynchronous operation
  for (auto *in : inputs_) {
    auto &p = static_cast<VarHandle *>(in)->place_;
    in->generated_op_->Wait(dev_ctxes_[p]);
  }

  // sum
  Variable var;
  ParameterCollection::Instance().Get(var_name)->Receive<Variable>(&var);

  Tensor reducer;
  Tensor temp;
  if (var.IsType<framework::SelectedRows>()) {
    // reduce sparse parameter

  } else if (var.IsType<framework::LoDTensor>()) {
    auto param = var.Get<LoDTensor>();

    auto dev_id = static_cast<platform::CUDAPlace>(param.place()).device;
    auto &p = nccl_ctxs_.DevCtx(dev_id);

    reducer.mutable_data(param.dims(), platform::CUDAPinnedPlace());
    temp.mutable_data(param.dims(), platform::CUDAPinnedPlace());

    framework::TensorCopy(param, platform::CUDAPinnedPlace(), dev_ctxes_[p],
                          reducer);
    dev_ctxes_[p]->Wait();
  } else {
    PADDLE_THROW("Parameter should be LoDTensor or SelectedRows");
  }

  // TODO(zcd): float should be T
  float *reducer_ptr = reducer.data<float>();
  for (int j = 0; j < device_count_ - 1; ++j) {
    Variable other_var;
    ParameterCollection::Instance().Get(var_name)->Receive<Variable>(
        &other_var);
    PADDLE_ENFORCE(other_var.Type() == var.Type());

    if (var.second->IsType<framework::SelectedRows>()) {
      // reduce sparse parameter

    } else if (var.second->IsType<framework::LoDTensor>()) {
      auto param = other_var.Get<LoDTensor>();
      PADDLE_ENFORCE_EQ(reducer.numel(), param.numel());

      auto dev_id = static_cast<platform::CUDAPlace>(param.place()).device;
      auto &p = nccl_ctxs_.DevCtx(dev_id);

      framework::TensorCopy(param, platform::CUDAPinnedPlace(), dev_ctxes_[p],
                            temp);

      dev_ctxes_[p]->Wait();
      float *temp_ptr = temp.data<float>();
      for (int k = 0; k < reducer.numel(); ++k) {
        reducer_ptr[k] += temp_ptr[k];
      }
    }
  }

  // broadcast
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    auto &p = places_[i];
    auto *s = local_scopes_[i];
    int dev_id = boost::get<platform::CUDAPlace>(p).device;
    auto var = s->FindVar(var_name);
    if (var.IsType<framework::SelectedRows>()) {
      // reduce sparse parameter

    } else if (var.IsType<framework::LoDTensor>()) {
      auto &lod_tensor = var->Get<LoDTensor>();
      void *buffer = const_cast<void *>(lod_tensor.data<void>());
      if (numel == 0) {
        numel = static_cast<size_t>(lod_tensor.numel());
      }

      auto dev_ctx = nccl_ctxs_.DevCtx(dev_id);
      framework::TensorCopy(reducer, p, dev_ctx, lod_tensor);
    }
  }
}

std::string AllReduceOpHandle::Name() const { return "all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
