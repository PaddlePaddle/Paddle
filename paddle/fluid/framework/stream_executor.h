// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <memory>
#include "operator.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cuda_api.h"

namespace paddle {
namespace framework {

using platform::CudaAPI;

struct ParallelMeta {
  int stream_id;
  platform::CudaAPI::stream_t stream;
};

class StreamOperation {
 public:
  StreamOperation(std::unique_ptr<OperatorBase>&& op, Scope* scope,
                  platform::Place place);

  void SetStream(platform::CudaAPI::stream_t stream);
  void SetInputEvents(std::vector<CudaAPI::event_t*> events);

  void CreateOperations();

  void Run() { op_->Run(*scope_, place_); }

  void AsyncRun() {
    if (dynamic_cast<OperatorWithKernel*>(op_.get())) {
      if (!runtime_context_) {
        runtime_context_.reset(
            new RuntimeContext(op_->Inputs(), op_->Outputs(), *scope_));
      }
    }
  }

 private:
  framework::Scope* scope_;
  platform::Place place_;
  std::unique_ptr<OperatorBase> op_;
  std::unique_ptr<RuntimeContext> runtime_context_;
  std::unique_ptr<RuntimeInferShapeContext> runtime_infershape_context_;
  std::unique_ptr < CUDADeviceContext
};

class StreamExecutor final {
 public:
  void Run();

 private:
  std::unique_ptr<ParallelMeta> parallel_meta_;
};

}  // namespace framework
}  // namespace paddle

#endif
