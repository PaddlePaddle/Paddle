/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include <nccl.h>
#endif

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace paddle {
namespace operators {

class CSyncCalcStreamOp : public framework::OperatorBase {
 public:
  CSyncCalcStreamOp(const std::string& type,
                    const framework::VariableNameMap& inputs,
                    const framework::VariableNameMap& outputs,
                    const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE(is_gpu_place(place),
                   "Sync stream op can run on gpu place only for now.");
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    cudaError_t e_sync = cudaStreamSynchronize(dev_ctx->stream());
    if (e_sync != 0) {
      LOG(FATAL) << "Fail to sync cuda stream: " << cudaGetErrorString(e_sync);
    }
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
};

class CSyncCalcStreamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync");
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync");
    AddComment(R"DOC(
CSyncCalcStream Operator

Call calculation stream synchronization.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_sync_calc_stream, ops::CSyncCalcStreamOp,
                  ops::CSyncCalcStreamOpMaker);
