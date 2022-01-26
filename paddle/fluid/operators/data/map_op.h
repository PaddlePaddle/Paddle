/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/data/map_runner.h"

namespace paddle {
namespace operators {

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;


static void CheckInputQueueStatus(const std::vector<Variable*>& vars) {
  for (auto var : vars) {
    PADDLE_ENFORCE_EQ(var->IsType<LoDTensorBlockingQueueHolder>(), true,
        platform::errors::InvalidArgument(
          "Input Variables of MapOp should hold "
          "LoDTensorBlockingQueueHolder type"));
    auto queue = var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
    PADDLE_ENFORCE_NE(queue, nullptr,
        platform::errors::InvalidArgument(
          "Input LoDTensorBlockingQueue is not initialized"));
  }
}

static void CheckAndInitOutputQueue(const std::vector<Variable*>& vars, int capacity) {
  for (auto var : vars) {
    if (var->IsInitialized()) {
      PADDLE_ENFORCE_EQ(var->IsType<LoDTensorBlockingQueueHolder>(), true,
          platform::errors::InvalidArgument(
            "Output Variables of MapOp should hold "
            "LoDTensorBlockingQueueHolder type"));
      auto queue = var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
      if (queue == nullptr) {
        auto* holder = var->template GetMutable<LoDTensorBlockingQueueHolder>();
        holder->InitOnce(capacity);
        VLOG(1) << "MapOpKernel init queue" << holder->GetQueue();
      }
    } else {
      VLOG(1) << "Initialize Output LoDTensorBlockingQueue capacity " << capacity;
      auto* holder = var->GetMutable<LoDTensorBlockingQueueHolder>();
      holder->InitOnce(capacity);
    }
  }
}

static std::vector<std::shared_ptr<LoDTensorBlockingQueue>> GetQueueVecFromVariableVec(const std::vector<Variable*>& vars) {
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> queues;
  queues.reserve(vars.size());
  for (size_t i = 0; i < vars.size(); i++) {
    queues.push_back(vars[i]->Get<LoDTensorBlockingQueueHolder>().GetQueue());
  }
  return queues;
}

template <typename DeviceContext, typename T>
class MapOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
