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

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {
namespace details {

class EarlyDeleteOpHandle : public OpHandleBase {
 public:
  EarlyDeleteOpHandle(ir::Node* node, const Scope* scope,
                      const platform::Place& place,
                      const std::vector<std::string>& names,
                      GarbageCollector* gc)
      : OpHandleBase(node),
        scope_(scope),
        place_(place),
        names_(names),
        gc_(gc) {
#ifdef PADDLE_WITH_CUDA
    if (IsStreamGarabageCollector()) {
      auto gpu_place = boost::get<platform::CUDAPlace>(place);
      PADDLE_ENFORCE(cudaSetDevice(gpu_place.device));
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
#endif
  }
  ~EarlyDeleteOpHandle() {
#ifdef PADDLE_WITH_CUDA
    if (IsStreamGarabageCollector()) {
      auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace());
      PADDLE_ENFORCE(cudaSetDevice(gpu_place.device));
      PADDLE_ENFORCE(cudaEventDestroy(event_));
    }
#endif
  }

  std::string Name() const override { return "early_delete"; }

 protected:
  void RunImpl() override {
    std::vector<std::shared_ptr<memory::Allocation>> tensors;
    auto* local_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope*>();
    for (auto& var_name : names_) {
      auto* var = local_scope->FindVar(var_name);
      PADDLE_ENFORCE(var != nullptr,
                     string::Sprintf("Local Scope not has var %s", var_name));
      if (var->IsType<LoDTensor>()) {
        tensors.emplace_back(var->GetMutable<LoDTensor>()->MoveMemoryHolder());
      } else if (var->IsType<SelectedRows>()) {
        tensors.emplace_back(var->GetMutable<SelectedRows>()
                                 ->mutable_value()
                                 ->MoveMemoryHolder());
      } else if (var->IsType<LoDTensorArray>()) {
        LoDTensorArray* tensor_array = var->GetMutable<LoDTensorArray>();
        for (auto& tensor : *tensor_array) {
          tensors.emplace_back(tensor.MoveMemoryHolder());
        }
      }
    }
    if (!tensors.empty()) {
      ClearTensors(tensors);
    }
  }

 private:
  void ClearTensors(
      const std::vector<std::shared_ptr<memory::Allocation>>& tensors) {
    if (platform::is_cpu_place(place_)) {
      ClearCPUTensors(tensors);
    } else {
      ClearGPUTensors(tensors);
    }
  }

  void ClearCPUTensors(
      const std::vector<std::shared_ptr<memory::Allocation>>& tensors) {
    auto* gc = dynamic_cast<CPUGarbageCollector*>(gc_);
    if (gc != nullptr) {
      gc->Add(tensors);
    }
  }

  void ClearGPUTensors(
      const std::vector<std::shared_ptr<memory::Allocation>>& tensors) {
#ifdef PADDLE_WITH_CUDA
    auto* gc = dynamic_cast<StreamGarbageCollector*>(gc_);
    if (gc != nullptr) {
      auto compute_stream = dev_ctx_->stream();
      auto callback_stream = gc->stream();
      auto callback_func = [=]() {
        PADDLE_ENFORCE(cudaEventRecord(event_, compute_stream));
        PADDLE_ENFORCE(cudaStreamWaitEvent(callback_stream, event_, 0));
      };
      gc_->Add(tensors, callback_func);
    } else {
      gc_->Add(tensors);
    }
  }

  bool IsStreamGarabageCollector() const {
    return dynamic_cast<const StreamGarbageCollector*>(gc_) != nullptr;
#endif
  }

  const Scope* scope_;
  const platform::Place place_;
  std::vector<std::string> names_;
  GarbageCollector* gc_;
#ifdef PADDLE_WITH_CUDA
  platform::CUDADeviceContext* dev_ctx_;
  cudaEvent_t event_;
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
