// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

namespace paddle {
namespace framework {
namespace details {

EagerDeletionOpHandle::EagerDeletionOpHandle(
    ir::Node *node, const Scope *scope, const platform::Place &place,
    const std::unordered_set<std::string> &var_names,
    GarbageCollector<Tensor> *gc, AtomicReferenceCountMap *ref_cnts)
    : OpHandleBase(node),
      scope_(scope),
      var_names_(var_names),
      gc_(gc),
      ref_cnts_(ref_cnts) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    dev_ctx_ = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    if (dynamic_cast<StreamGarbageCollector<Tensor> *>(gc_)) {
      platform::CUDADeviceGuard guard(
          boost::get<platform::CUDAPlace>(place).device);
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
      PADDLE_ENFORCE_NOT_NULL(event_);
    }
  }
#endif
}

EagerDeletionOpHandle::~EagerDeletionOpHandle() {
#ifdef PADDLE_WITH_CUDA
  if (event_) {
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace());
    platform::CUDADeviceGuard guard(gpu_place.device);
    PADDLE_ENFORCE(cudaEventDestroy(event_));
  }
#endif
}

std::string EagerDeletionOpHandle::Name() const { return "eager_deletion"; }

void EagerDeletionOpHandle::RunImpl() {
  auto *exec_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
  std::vector<Tensor *> tensors;
  for (auto &name : var_names_) {
    auto it = ref_cnts_->find(name);
    if (it == ref_cnts_->end()) {
      continue;
    }

    auto *var = exec_scope->FindVar(name);
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<LoDTensor>()) {
      if (it->second.fetch_sub(1) == 1) {
        tensors.emplace_back(var->GetMutable<LoDTensor>());
      }
    } else if (var->IsType<SelectedRows>()) {
      if (it->second.fetch_sub(1) == 1) {
        tensors.emplace_back(var->GetMutable<SelectedRows>()->mutable_value());
      }
    } else if (var->IsType<LoDTensorArray>()) {
      if (it->second.fetch_sub(1) == 1) {
        auto *tensor_arr = var->GetMutable<LoDTensorArray>();
        for (auto &t : *tensor_arr) {
          tensors.emplace_back(&t);
        }
      }
    }
  }

  if (!tensors.empty()) {
    ClearTensors(tensors);
  }
}

void EagerDeletionOpHandle::ClearTensors(const std::vector<Tensor *> &tensors) {
#ifdef PADDLE_WITH_CUDA
  if (event_) {
    auto compute_stream = dev_ctx_->stream();
    auto callback_stream =
        static_cast<StreamGarbageCollector<Tensor> *>(gc_)->stream();
    auto callback_func = [=]() {
      PADDLE_ENFORCE(cudaEventRecord(event_, compute_stream));
      PADDLE_ENFORCE(cudaStreamWaitEvent(callback_stream, event_, 0));
    };
    gc_->Add(tensors, callback_func);
  } else {
#endif
    gc_->Add(tensors);
#ifdef PADDLE_WITH_CUDA
  }
#endif
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
