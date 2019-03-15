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

#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/container_cast.h"

namespace paddle {
namespace framework {
namespace details {

FetchOpHandle::FetchOpHandle(ir::Node *node, FeedFetchList *data, size_t offset,
                             std::vector<Scope *> *local_scopes)
    : OpHandleBase(node),
      data_(data),
      offset_(offset),
      local_scopes_(local_scopes) {}

FetchOpHandle::~FetchOpHandle() {}

void FetchOpHandle::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
  PADDLE_THROW("Nobody should wait FetchOp. Unexpceted Error");
}

void FetchOpHandle::WaitAndMergeCPUTensors() const {
  std::vector<const LoDTensor *> tensors_ptr;
  tensors_ptr.reserve(tensors_.size());
  for (auto &t : tensors_) {
    tensors_ptr.emplace_back(&t);
  }
  data_->at(offset_).MergeLoDTensor(tensors_ptr, platform::CPUPlace());
}

void FetchOpHandle::RunImpl() {
  auto cpu_ctx =
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  RecordWaitEventOnCtx2(in_var_handles, cpu_ctx);

  tensors_.resize(inputs_.size());
  platform::CPUPlace cpu;
  auto &scopes = *local_scopes_;

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto *var_handle = static_cast<VarHandle *>(inputs_[i]);
    auto &scope = scopes.at(var_handle->scope_idx());
    auto *var = scope->FindVar(kLocalExecScopeName)
                    ->Get<Scope *>()
                    ->FindVar(var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(var, "Cannot find variable %s in execution scope",
                            var_handle->name());

    auto &t = var->Get<framework::LoDTensor>();
    if (platform::is_gpu_place(t.place())) {
#ifdef PADDLE_WITH_CUDA
      TensorCopySync(t, cpu, &tensors_[i]);
#endif
    } else {
      tensors_[i].ShareDataWith(t);
    }
    tensors_[i].set_lod(t.lod());
  }

  this->WaitAndMergeCPUTensors();
}

void FetchOpHandle::WaitInputVarGenerated(const platform::Place &place) {
  auto cpu_ctx = platform::DeviceContextPool::Instance().Get(place);
  for (auto *input : inputs_) {
    if (input->GeneratedOp()) {
      input->GeneratedOp()->RecordWaitEventOnCtx(cpu_ctx);
    }
  }
}

std::string FetchOpHandle::Name() const { return "Fetch"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
