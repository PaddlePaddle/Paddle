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

#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif
#include <algorithm>

namespace paddle::framework {
class Variable;
}  // namespace paddle::framework

namespace paddle::framework::details {

EagerDeletionOpHandle::EagerDeletionOpHandle(
    ir::Node *node,
    Scope *scope,
    size_t scope_idx,
    const phi::Place &place,
    const std::unordered_set<ir::MemOptVarInfo *> &vars,
    GarbageCollector *gc)
    : OpHandleBase(node),
      scope_(scope),
      scope_idx_(scope_idx),
      place_(place),
      var_infos_(vars.begin(), vars.end()),
      gc_(gc),
      vars_() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_gpu_place(place)) {
    dev_ctx_ = reinterpret_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(place));
    if (dynamic_cast<StreamGarbageCollector *>(gc_)) {
      platform::CUDADeviceGuard guard(place.device);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipEventCreateWithFlags(&event_, hipEventDisableTiming));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
#endif
      PADDLE_ENFORCE_NOT_NULL(
          event_,
          common::errors::InvalidArgument("The cuda event created is NULL."));
    }
  }
#endif
  PADDLE_ENFORCE_NE(vars.empty(),
                    true,
                    common::errors::InvalidArgument(
                        "The variables to be deleted are empty."));
  for (auto *var : var_infos_) {
    PADDLE_ENFORCE_NOT_NULL(var,
                            common::errors::InvalidArgument(
                                "The memory optimization info is NULL."));
  }
}

EagerDeletionOpHandle::~EagerDeletionOpHandle() {  // NOLINT
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (event_) {
    auto gpu_place = dev_ctx_->GetPlace();
    platform::CUDADeviceGuard guard(gpu_place.device);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(event_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event_));
#endif
  }
#endif
}

void EagerDeletionOpHandle::InitCUDA() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  int dev_id = dev_ctxes_.begin()->first.device;
  events_[dev_id] = nullptr;
#endif
}

void EagerDeletionOpHandle::CallOnce() {
  PADDLE_ENFORCE_EQ(
      vars_.empty(),
      true,
      common::errors::InvalidArgument(
          "The variables to be deleted should be initialized here."));
  Scope *exec_scope = local_exec_scopes_[0];
  for (auto *var_info : var_infos_) {
    auto *var = exec_scope->FindVar(var_info->Name());
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound(
            "The variable(%s) to be inplaced is not found in scope.",
            var_info->Name()));
    vars_.emplace_back(var);
  }
}

std::string EagerDeletionOpHandle::Name() const { return "eager_deletion"; }

static bool CanBeErased(ir::MemOptVarInfo *var_info) {
  if (var_info->IsSkippedAllMemoryOptimization() ||
      !var_info->DecreaseRefCnt()) {
    return false;
  }
#ifdef PADDLE_WITH_CINN
  // if parent_holder exists, it should meet deletion condition too.
  std::shared_ptr<ir::MemOptVarInfo> parent_holder = var_info->ParentHolder();
  if (parent_holder && !CanBeErased(parent_holder.get())) {
    return false;
  }
#endif
  return true;
}

void EagerDeletionOpHandle::RunImpl() {
  if (vars_.size() != var_infos_.size() || is_variant_scope_) {
    vars_.clear();
    CallOnce();
  }

  phi::RecordEvent record_event(Name(), phi::TracerEventType::UserDefined, 2);
  std::deque<std::shared_ptr<memory::Allocation>> garbages;
  for (size_t i = 0; i < var_infos_.size(); ++i) {
    auto *var_info = var_infos_[i];
    if (!CanBeErased(var_info)) {
      VLOG(4) << "skip memory optimization with var: " << var_info->Name();
      continue;
    }

    VLOG(2) << "Erase variable " << var_info->Name() << " on " << place_;

    Variable *var = vars_[i];

    if (var->IsType<phi::DenseTensor>()) {
      garbages.emplace_back(
          var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
    } else if (var->IsType<phi::SelectedRows>()) {
      garbages.emplace_back(var->GetMutable<phi::SelectedRows>()
                                ->mutable_value()
                                ->MoveMemoryHolder());
    } else if (var->IsType<phi::TensorArray>()) {
      auto *tensor_arr = var->GetMutable<phi::TensorArray>();
      for (auto &t : *tensor_arr) {
        garbages.emplace_back(t.MoveMemoryHolder());
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "The variable(%s) of type %s is not supported in eager deletion.",
          framework::ToTypeName(var->Type()),
          var_info->Name()));
    }
  }

  if (!garbages.empty()) {
    ClearGarbages(&garbages);
  }
}

void EagerDeletionOpHandle::ClearGarbages(
    std::deque<std::shared_ptr<memory::Allocation>> *garbages) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (event_) {
    auto compute_stream = dev_ctx_->stream();
    auto callback_stream =
        reinterpret_cast<StreamGarbageCollector *>(gc_)->stream();
    auto callback_func = [=]() {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event_, compute_stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamWaitEvent(callback_stream, event_, 0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_, compute_stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamWaitEvent(callback_stream, event_, 0));
#endif
    };
    gc_->Add(std::move(*garbages), callback_func);
  } else {
#endif
    gc_->Add(std::move(*garbages));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif
}

std::vector<std::string> EagerDeletionOpHandle::VarsToDelete() const {
  std::vector<std::string> var_names;
  var_names.reserve(var_infos_.size());
  for (auto &info : var_infos_) {
    var_names.emplace_back(info->Name());
  }
  std::sort(var_names.begin(), var_names.end());
  return var_names;
}

}  // namespace paddle::framework::details
