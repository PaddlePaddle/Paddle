/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SendAndRecvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& scope = ctx.scope();
    const auto& place = ctx.GetPlace();
    auto message_name = ctx.Attr<std::string>("message_name");
    auto send_var_name = ctx.Attr<std::vector<std::string>>("send_var_name");
    auto recv_var_name = ctx.Attr<std::vector<std::string>>("recv_var_name");
    auto next_epmap = ctx.Attr<std::vector<std::string>>("next_endpoints");
    auto previous_epmap =
        ctx.Attr<std::vector<std::string>>("previous_endpoints");
    auto trainer_id = ctx.Attr<int>("trainer_id");
    auto mode = ctx.Attr<std::string>("mode");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& context = *pool.Get(place);

    distributed::HeterClient* rpc_client =
        distributed::HeterClient::GetInstance(next_epmap, previous_epmap,
                                              trainer_id)
            .get();
    VLOG(3) << "SendAndRecvOp message_name: " << message_name;
    rpc_client->SendAndRecvAsync(context, scope, message_name, send_var_name,
                                 recv_var_name, mode);
  }
};

class SendAndRecvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, platform::CPUPlace());
  }
};

class SendAndRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "Tensor Input variable to be sent").AsDuplicable();
    AddOutput("Out", "Tensor Output varibale to be recv").AsDuplicable();
    AddAttr<std::string>("message_name", "");
    AddAttr<std::string>("mode", "forward or backward").SetDefault("forward");
    AddAttr<std::vector<std::string>>("send_var_name", "Send Tensor's name");
    AddAttr<std::vector<std::string>>("recv_var_name", "Recv Tensor's name");
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<std::string>>("endpoints", "Server endpoint")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<std::vector<std::string>>("next_endpoints", "Server endpoint")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<std::vector<std::string>>("previous_endpoints",
                                      "Previous Server endpoint")
        .SetDefault({"127.0.0.1:6164"});
    AddComment(R"DOC(
    SendAndRecv operator
    This operator will send variables to listen_and_serve op at the parameter server.
    And recv variable from parameter server of send variable's scope.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send_and_recv, ops::SendAndRecvOp, ops::SendAndRecvOpMaker);
REGISTER_OP_CUDA_KERNEL(
    send_and_recv,
    ops::SendAndRecvKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SendAndRecvKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SendAndRecvKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SendAndRecvKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    send_and_recv,
    ops::SendAndRecvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SendAndRecvKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SendAndRecvKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SendAndRecvKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_VERSION(send_and_recv)
    .AddCheckpoint(
        R"ROC(add new attributes [next_endpoints] [previous_endpoints] and [mode])ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("next_endpoints", "Server endpoint",
                     std::vector<std::string>({"127.0.0.1:6164"}))
            .NewAttr("previous_endpoints", "Server endpoint",
                     std::vector<std::string>({"127.0.0.1:6164"}))
            .NewAttr("mode", "forward or backward", "forward"));
