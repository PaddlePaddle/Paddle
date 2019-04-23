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

#include <memory>
#include <unordered_set>
#include <utility>

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
    const std::unordered_set<std::string> &var_names, GarbageCollector *gc,
    AtomicReferenceCountMap *ref_cnts)
    : OpHandleBase(node),
      scope_(scope),
      var_names_(var_names.begin(), var_names.end()),
      gc_(gc),
      ref_cnts_(ref_cnts) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    dev_ctx_ = reinterpret_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    if (dynamic_cast<StreamGarbageCollector *>(gc_)) {
      platform::CUDADeviceGuard guard(
          boost::get<platform::CUDAPlace>(place).device);
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
      PADDLE_ENFORCE_NOT_NULL(event_);
    }
  }
#endif
  PADDLE_ENFORCE(!var_names_.empty(), "Var names cannot be empty");
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
  Scope *exec_scope = nullptr;
  std::deque<std::shared_ptr<memory::Allocation>> garbages;
  for (auto &name : var_names_) {
    auto it = ref_cnts_->find(name);
    // Reference count has not decreased to 0
    if (it == ref_cnts_->end() || it->second.fetch_sub(1) != 1) {
      continue;
    }

    if (!exec_scope) {
      exec_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
    }

    // Var not found
    auto *var = exec_scope->FindVar(name);
    if (var == nullptr) {
      continue;
    }

    VLOG(2) << "Erase variable " << name;

    if (var->IsType<LoDTensor>()) {
      garbages.emplace_back(var->GetMutable<LoDTensor>()->MoveMemoryHolder());
    } else if (var->IsType<SelectedRows>()) {
      garbages.emplace_back(
          var->GetMutable<SelectedRows>()->mutable_value()->MoveMemoryHolder());
    } else if (var->IsType<LoDTensorArray>()) {
      auto *tensor_arr = var->GetMutable<LoDTensorArray>();
      for (auto &t : *tensor_arr) {
        garbages.emplace_back(t.MoveMemoryHolder());
      }
    } else {
      PADDLE_THROW("Type %s of %s is not supported eager deletion",
                   framework::ToTypeName(var->Type()), name);
    }
  }

  if (!garbages.empty()) {
    ClearGarbages(&garbages);
  }
}

void EagerDeletionOpHandle::ClearGarbages(
    std::deque<std::shared_ptr<memory::Allocation>> *garbages) {
#ifdef PADDLE_WITH_CUDA
  if (event_) {
    auto compute_stream = dev_ctx_->stream();
    auto callback_stream =
        reinterpret_cast<StreamGarbageCollector *>(gc_)->stream();
    auto callback_func = [=]() {
      PADDLE_ENFORCE(cudaEventRecord(event_, compute_stream));
      PADDLE_ENFORCE(cudaStreamWaitEvent(callback_stream, event_, 0));
    };
    gc_->Add(std::move(*garbages), callback_func);
  } else {
#endif
    gc_->Add(std::move(*garbages));
#ifdef PADDLE_WITH_CUDA
  }
#endif
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
