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
#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/reference_count_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

using details::UnlivedNodePool;

class EarlyDeleteOpHandle : public OpHandleBase {
 public:
  EarlyDeleteOpHandle(ir::Node* node, const Scope* scope,
                      const platform::CUDAPlace& place,
                      const std::vector<std::string>& names,
                      GarbageCollector<Tensor>* gc)
      : OpHandleBase(node), scope_(scope), names_(names), gc_(gc) {
    if (IsStreamGarabageCollector()) {
      PADDLE_ENFORCE(cudaSetDevice(place.device));
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
  }
  ~EarlyDeleteOpHandle() {
    if (IsStreamGarabageCollector()) {
      auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace());
      PADDLE_ENFORCE(cudaSetDevice(gpu_place.device));
      PADDLE_ENFORCE(cudaEventDestroy(event_));
    }
  }

  std::string Name() const override { return "early_delete"; }

 protected:
  void RunImpl() override {
    auto* local_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope*>();
    std::vector<Tensor*> tensors;
    for (auto& var_name : names_) {
      auto* var = local_scope->FindVar(var_name);
      PADDLE_ENFORCE(var != nullptr,
                     string::Sprintf("Local Scope not has var %s", var_name));
      if (var->IsType<LoDTensor>()) {
        tensors.emplace_back(var->GetMutable<LoDTensor>());
      } else if (var->IsType<SelectedRows>()) {
        tensors.emplace_back(var->GetMutable<SelectedRows>()->mutable_value());
      } else if (var->IsType<LoDTensorArray>()) {
        LoDTensorArray* tensor_array = var->GetMutable<LoDTensorArray>();
        for (auto& tensor : *tensor_array) {
          tensors.emplace_back(static_cast<Tensor*>(&tensor));
        }
      }
    }
    if (!tensors.empty()) {
      ClearTensors(tensors);
    }
  }

 private:
  void ClearTensors(const std::vector<Tensor*>& tensors) {
    auto* gc = dynamic_cast<StreamGarbageCollector<Tensor>*>(gc_);
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
    return dynamic_cast<const StreamGarbageCollector<Tensor>*>(gc_) != nullptr;
  }

  const Scope* scope_;
  platform::CUDADeviceContext* dev_ctx_;
  std::vector<std::string> names_;
  GarbageCollector<Tensor>* gc_;
  cudaEvent_t event_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
