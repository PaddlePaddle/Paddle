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

#include <atomic>
#include <queue>
#include <string>
#include <unordered_map>
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

using ReferenceCountMap = std::unordered_map<std::string, int>;
using AtomicReferenceCountMap =
    std::unordered_map<std::string, std::atomic<int>>;
using DeviceReferenceCountMap =
    std::unordered_map<int, std::unique_ptr<ReferenceCountMap>>;
using AtomicDeviceReferenceCountMap =
    std::unordered_map<int, std::unique_ptr<AtomicReferenceCountMap>>;
using DeviceGarbageCollectorMap =
    std::unordered_map<int,
                       std::unique_ptr<GarbageCollector<framework::Tensor>>>;
constexpr int kDefaultCPUGarbageCollectorId =
    1 << 8;  // avoid conflict with gpus.

class ReferenceCountOpHandle : public OpHandleBase {
 public:
  ReferenceCountOpHandle(ir::Node *node, const Scope *scope,
                         const platform::CUDAPlace &place,
                         const std::vector<std::string> &var_names,
                         GarbageCollector<Tensor> *gc,
                         AtomicReferenceCountMap *ref_cnts)
      : OpHandleBase(node), scope_(scope), gc_(gc), ref_cnts_(ref_cnts) {
    dev_ctx_ = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    if (IsStreamGarabageCollector()) {
      platform::SetDeviceId(place.device);
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }

    for (auto &name : var_names) AddVar(name);
  }

  ~ReferenceCountOpHandle() {
    if (IsStreamGarabageCollector()) {
      auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace());
      platform::SetDeviceId(gpu_place.device);
      PADDLE_ENFORCE(cudaEventDestroy(event_));
    }
  }

  std::string Name() const override { return "reference_count"; }

  void AddVar(const std::string &name) {
    auto it = var_names_.find(name);
    if (it != var_names_.end())
      ++(it->second);
    else
      var_names_[name] = 1;
  }

 protected:
  void RunImpl() override {
    auto *exec_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
    std::vector<Tensor *> tensors;
    for (auto &pair : var_names_) {
      auto &name = pair.first;
      auto it = ref_cnts_->find(name);
      if (it == ref_cnts_->end()) continue;

      auto *var = exec_scope->FindVar(name);
      if (var == nullptr) continue;

      if (var->IsType<LoDTensor>()) {
        if (it->second.fetch_sub(pair.second) <= pair.second) {
          tensors.emplace_back(var->GetMutable<LoDTensor>());
        }
      } else if (var->IsType<SelectedRows>()) {
        if (it->second.fetch_sub(pair.second) <= pair.second) {
          tensors.emplace_back(
              var->GetMutable<SelectedRows>()->mutable_value());
        }
      } else if (var->IsType<LoDTensorArray>()) {
        if (it->second.fetch_sub(pair.second) <= pair.second) {
          LoDTensorArray *tensor_array = var->GetMutable<LoDTensorArray>();
          for (auto &tensor : *tensor_array) {
            tensors.emplace_back(static_cast<Tensor *>(&tensor));
          }
        }
      }
    }

    if (!tensors.empty()) {
      ClearTensors(tensors);
    }
  }

 private:
  void ClearTensors(const std::vector<Tensor *> &tensors) {
    auto *gc = dynamic_cast<StreamGarbageCollector<Tensor> *>(gc_);
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
    return dynamic_cast<const StreamGarbageCollector<Tensor> *>(gc_) != nullptr;
  }

  const Scope *scope_;
  platform::CUDADeviceContext *dev_ctx_;
  std::unordered_map<std::string, int> var_names_;
  GarbageCollector<Tensor> *gc_;       // not own
  AtomicReferenceCountMap *ref_cnts_;  // not own
  cudaEvent_t event_;
};

static ComputationOpHandle *FindNextComputationOpHandle(VarHandle *var_in) {
  std::queue<VarHandleBase *> queue;
  queue.push(var_in);
  do {
    auto *var = queue.front();
    queue.pop();
    for (auto *op : var->PendingOps()) {
      auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
      if (compute_op != nullptr && compute_op->GetPlace() == var_in->place_) {
        return compute_op;
      }
      for (auto *out_var : op->Outputs()) {
        queue.push(out_var);
      }
    }
  } while (!queue.empty());
  return nullptr;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
