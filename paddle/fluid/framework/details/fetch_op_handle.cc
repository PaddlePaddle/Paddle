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
#include <utility>
#include <vector>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

FetchOpHandle::FetchOpHandle(ir::Node *node, FetchResultType *data,
                             size_t offset, std::vector<Scope *> *local_scopes,
                             std::vector<Scope *> *local_exec_scopes,
                             bool return_merged)
    : OpHandleBase(node),
      data_(data),
      offset_(offset),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      return_merged_(return_merged) {}

FetchOpHandle::~FetchOpHandle() {}

void FetchOpHandle::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
  PADDLE_THROW("Nobody should wait FetchOp. Unexpceted Error");
}

void FetchOpHandle::WaitAndMergeCPUTensors() const {
  if (return_merged_) {
    const auto &tensor_dims = tensors_[0].dims();
    for (size_t i = 1; i < tensors_.size(); i++) {
      const auto &ele_dims = tensors_[i].dims();
      PADDLE_ENFORCE_EQ(
          tensor_dims.size(), ele_dims.size(),
          platform::errors::Fatal("The dimension sizes of fetched Tensors are "
                                  "different from each other on different "
                                  "devices. And the error is caused by the %zu "
                                  "(th) fetched variable. Please set the "
                                  "parameter `return_merged = False` when you "
                                  "call the `Executor.run()` method.",
                                  offset_));
      for (int j = 1; j < tensor_dims.size(); j++) {
        PADDLE_ENFORCE_EQ(
            tensor_dims[j], ele_dims[j],
            platform::errors::Fatal("The dimensions of fetched Tensors are "
                                    "different from each other on different "
                                    "devices. And the error is caused by the "
                                    "%zu (th) fetched variable. Please set the "
                                    "parameter `return_merged = False` when "
                                    "you call the `Executor.run()` method.",
                                    offset_));
      }
    }
    std::vector<const LoDTensor *> tensors_ptr;
    tensors_ptr.reserve(tensors_.size());
    for (auto &t : tensors_) {
      tensors_ptr.emplace_back(&t);
    }
    auto &val = boost::get<FeedFetchList>(*data_);
    val.at(offset_).MergeLoDTensor(tensors_ptr, platform::CPUPlace());
  } else {
    auto &val = boost::get<FetchUnmergedList>(*data_);
    val.at(offset_) = std::move(tensors_);
  }
}

void FetchOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());
  WaitInputVarGenerated(platform::CPUPlace());

  tensors_.resize(inputs_.size());
  platform::CPUPlace cpu;
  auto &scopes = *local_exec_scopes_;

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto *var_handle = static_cast<VarHandle *>(inputs_[i]);
    auto &scope = scopes.at(var_handle->scope_idx());
    auto *var = scope->FindVar(var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(var, "Cannot find variable %s in execution scope",
                            var_handle->name());

    auto &t = var->Get<framework::LoDTensor>();
    if (t.IsInitialized() && t.numel() > 0) {
      if (platform::is_gpu_place(t.place())) {
#ifdef PADDLE_WITH_CUDA
        TensorCopy(t, cpu, &tensors_[i]);
#endif
      } else {
        tensors_[i].ShareDataWith(t);
      }
    } else {
      tensors_[i].clear();
      tensors_[i].Resize({0});
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

bool FetchOpHandle::IsMultiDeviceTransfer() { return true; }

std::string FetchOpHandle::Name() const { return "Fetch"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
