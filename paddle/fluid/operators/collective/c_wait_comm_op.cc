/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <string>

#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace paddle {
namespace operators {

class CWaitCommOp : public framework::OperatorBase {
 public:
  CWaitCommOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "wait_comm op can run on gpu place only for now."));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    int ring_id = Attr<int>("ring_id");

    auto compute_stream =
        static_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place))
            ->stream();
    auto comm_stream =
        platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();

    auto event =
        platform::NCCLCommContext::Instance().Get(ring_id, place)->comm_event();

// comm_stream-->event-->compute_stream
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(event, comm_stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamWaitEvent(compute_stream, event, 0));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, comm_stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(compute_stream, event, 0));
#endif
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

class CWaitCommOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddComment(R"DOC(
CWaitComm Operator

Compute stream wait Comm Stream with async event.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_wait_comm, ops::CWaitCommOp, ops::CWaitCommOpMaker);
