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

#include "paddle/fluid/operators/swap_mem_op.h"
#include <cuda.h>
#include <string>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class SwapMemCPUToGPUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SwapCPUtoGPU");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    // DO NOT Transform Device
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class SwapMemGPUToCPUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SwapCPUtoGPU");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    // DO NOT Transform Device
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class SwapMemCPUToGPUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of swapmem op");
    AddOutput("Out", "The output of SwapMemCPUToGPU Op.");
    AddComment(R"DOC(
      SwapMemCPUToGPU Operator.
      )DOC");
  }
};

class SwapMemGPUToCPUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of swapmem op");
    AddOutput("Out", "The output of SwapMemGPUToCPU Op.");
    AddComment(R"DOC(
      SwapMemGPUToCPU Operator.
      )DOC");
  }
};

template <typename DeviceContext, typename T>
class SwapMemCPUToGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // in is in CPU Place
    // auto* x = context.Input<Tensor>("X");
    auto* in = context.Input<Tensor>("X");
    PADDLE_ENFORCE_EQ(
        in->place(), platform::CPUPlace(),
        platform::errors::InvalidArgument(
            "Input of SwapMemCPUToGPU shoud be a variable in cpu place"));
    auto* in_data = in->data<T>();
    auto* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    auto stream = context.cuda_device_context().stream();
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()), out_data,
                 platform::CPUPlace(), in_data, sizeof(T) * in->numel(),
                 stream);
  }
};

template <typename DeviceContext, typename T>
class SwapMemGPUToCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // in is in CUDA Place
    // auto* x = context.Input<Tensor>("X");
    auto* in = context.Input<Tensor>("X");
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(in->place()), true,
        platform::errors::InvalidArgument(
            "Input of SwapMemGPUToCPU shoud be a variable in gpu place"));
    auto* in_data = in->data<T>();
    auto* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<T>(platform::CPUPlace());
    auto stream = context.cuda_device_context().stream();
    memory::Copy(platform::CPUPlace(), out_data,
                 boost::get<platform::CUDAPlace>(in->place()), in_data,
                 sizeof(T) * in->numel(), stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    swapmem_cpu2gpu, ops::SwapMemCPUToGPUOp, ops::SwapMemCPUToGPUOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CUDA_KERNEL(
    swapmem_cpu2gpu,
    ops::SwapMemCPUToGPUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SwapMemCPUToGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SwapMemCPUToGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SwapMemCPUToGPUKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OPERATOR(
    swapmem_gpu2cpu, ops::SwapMemGPUToCPUOp, ops::SwapMemGPUToCPUOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CUDA_KERNEL(
    swapmem_gpu2cpu,
    ops::SwapMemGPUToCPUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SwapMemGPUToCPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SwapMemGPUToCPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SwapMemGPUToCPUKernel<paddle::platform::CUDADeviceContext, double>);
