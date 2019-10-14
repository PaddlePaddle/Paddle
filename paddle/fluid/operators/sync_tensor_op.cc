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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SyncTensorOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");
    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(*in, context.GetPlace(), out);
  }
};

class SyncTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class SyncTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The source tensor.");
    AddOutput("Out", "(LoDTensor) The destination tensor.");
    AddComment(R"DOC(
SyncTensorToDevice Operator.

Synchronize the input tensor to the output tensor.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(sync_tensor, paddle::operators::SyncTensorOp,
                  paddle::operators::SyncTensorOpMaker);
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CPU_KERNEL(
    sync_tensor,
    ops::SyncTensorOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SyncTensorOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SyncTensorOpKernel<paddle::platform::CPUDeviceContext, double>);
#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    sync_tensor,
    ops::SyncTensorOpKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::SyncTensorOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SyncTensorOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SyncTensorOpKernel<paddle::platform::CUDADeviceContext, double>);
#endif
