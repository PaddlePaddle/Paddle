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
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace operators {

class CWaitCommOp : public framework::OperatorBase {
 public:
  CWaitCommOp(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& place) const override {
    PADDLE_ENFORCE_EQ(
        place.GetType() == phi::AllocationType::GPU,
        true,
        phi::errors::PreconditionNotMet(
            "wait_comm op can run on gpu place only for now, but got %s",
            place.DebugString()));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    int ring_id = Attr<int>("ring_id");

    gpuStream_t compute_stream =
        static_cast<phi::GPUContext*>(
            phi::DeviceContextPool::Instance().Get(place))
            ->stream();
    gpuStream_t comm_stream = nullptr;
    gpuEvent_t event = nullptr;

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                        true,
                        phi::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(ring_id)));
      phi::distributed::NCCLCommContext* comm_ctx =
          static_cast<phi::distributed::NCCLCommContext*>(
              comm_context_manager.Get(std::to_string(ring_id)));
      comm_stream = comm_ctx->GetStream();
      event = comm_ctx->GetComputeEvent();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {
      comm_stream =
          platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();

      event = platform::NCCLCommContext::Instance()
                  .Get(ring_id, place)
                  ->comm_event();
      VLOG(3) << "old NCCLCommContext has rid " << ring_id;
    }

// comm_stream-->event-->compute_stream
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, comm_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(compute_stream, event, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, comm_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(compute_stream, event, 0));
#endif
#else
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

class CWaitCommOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
