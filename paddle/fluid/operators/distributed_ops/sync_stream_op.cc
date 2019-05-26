/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <nccl.h>
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

class SyncStreamOp : public framework::OperatorBase {
 public:
  SyncStreamOp(const std::string& type,
               const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE(is_gpu_place(place),
                   "Sync stream op can run on gpu place only for now.");

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    int sync_type = Attr<int>("sync_type");
    if (sync_type == NO_SYNC) {
      return;
    }

    if (sync_type & SYNC_CUDA) {
      auto dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
      cudaError_t e_sync = cudaStreamSynchronize(dev_ctx->stream());
      if (e_sync != 0) {
        LOG(FATAL) << "Fail to sync cuda stream: "
          << cudaGetErrorString(e_sync);
      }
    }

    if (sync_type & SYNC_NCCL) {
      auto dev_ctx =
        platform::NCCLCommContext::Instance().DevCtx(place);
      cudaError_t e_sync = cudaStreamSynchronize(dev_ctx->stream());
      if (e_sync != 0) {
        LOG(FATAL) << "Fail to sync nccl stream: "
          << cudaGetErrorString(e_sync);
      }
    }
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }

 private:
  enum {
    NO_SYNC = 0,
    SYNC_CUDA = 1,
    SYNC_NCCL = 2,
  };
};

class SyncStreamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<int>("sync_type",
        "(int) bitwise attribute. "
        "0 (000): do not synchronize; "
        "1 (001): synchronize cuda stream; "
        "2 (010): synchronize nccl stream; ").SetDefault(0);
    AddComment(R"DOC(
***Sync Operator***

Call stream synchronize.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sync_stream, ops::SyncStreamOp, ops::SyncStreamOpMaker);
