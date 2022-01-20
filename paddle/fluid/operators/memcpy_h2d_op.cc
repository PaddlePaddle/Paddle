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

#include "paddle/fluid/operators/memcpy_h2d_op.h"

#include <string>

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class MemcpyH2DOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto type = ctx->GetInputsVarType("X")[0];
    if (type == framework::proto::VarType::SELECTED_ROWS ||
        type == framework::proto::VarType::LOD_TENSOR) {
      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
      if (type == framework::proto::VarType::LOD_TENSOR) {
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class MemcpyH2DInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class MemcpyH2DKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    if (x == nullptr) {
      return;
    }
    PADDLE_ENFORCE_EQ(ctx.HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of memcpy_d2h_op is not found."));
    auto *out = ctx.OutputVar("Out");
    // Get dev_ctx from ExecutionContext, it's H2D stream
    auto &dev_ctx = ctx.device_context();
    auto dst_place_type = ctx.Attr<int>("dst_place_type");
    framework::VisitVarType(*x, MemcpyH2DFunctor(out, dev_ctx, dst_place_type));
  }
};

class MemcpyH2DOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input variable ");
    AddOutput("Out",
              "(LoDTensor) The type of output "
              "is the same as input X.");
    AddAttr<int>(
        "dst_place_type",
        "Determine the dst place of tensor copy. "
        "By Now it ONLY support CUDAPinnedPlace/CPU <-> NPUPlace/CUDAPlace "
        "Other place type is Unimplemented and will cause ERROR."
        "0: dst is on CUDAPlace. "
        "1: dst is on NPUPlace. ");
    AddComment(R"DOC(
    MemcpyD2H Operator.
    By now, it ONLY supports the memcopy between CUDAPinnedPlace/CPU <-> NPUPlace/CUDAPlace.
    You would have to update it if you want other more capacities.
Out = X,  when type in [LoDTensor]
raise error if the type is not listed above.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    memcpy_h2d, ops::MemcpyH2DOp, ops::MemcpyH2DOpProtoMaker,
    ops::MemcpyH2DInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL_FUNCTOR(
    memcpy_h2d, float, ops::MemcpyH2DKernel, double, ops::MemcpyH2DKernel,
    int8_t, ops::MemcpyH2DKernel, uint8_t, ops::MemcpyH2DKernel, int,
    ops::MemcpyH2DKernel, int64_t, ops::MemcpyH2DKernel, bool,
    ops::MemcpyH2DKernel, paddle::platform::bfloat16, ops::MemcpyH2DKernel,
    paddle::platform::complex<float>, ops::MemcpyH2DKernel,
    paddle::platform::complex<double>, ops::MemcpyH2DKernel, plat::float16,
    ops::MemcpyH2DKernel, int16_t, ops::MemcpyH2DKernel);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL_FUNCTOR(
    memcpy_h2d, float, ops::MemcpyH2DKernel, double, ops::MemcpyH2DKernel,
    int8_t, ops::MemcpyH2DKernel, uint8_t, ops::MemcpyH2DKernel, int,
    ops::MemcpyH2DKernel, int64_t, ops::MemcpyH2DKernel, bool,
    ops::MemcpyH2DKernel, paddle::platform::bfloat16, ops::MemcpyH2DKernel,
    paddle::platform::complex<float>, ops::MemcpyH2DKernel,
    paddle::platform::complex<double>, ops::MemcpyH2DKernel, plat::float16,
    ops::MemcpyH2DKernel, int16_t, ops::MemcpyH2DKernel);
#endif

#ifdef PADDLE_WITH_ASCEND_CL
REGISTER_OP_NPU_KERNEL_FUNCTOR(
    memcpy_h2d, float, ops::MemcpyH2DKernel, double, ops::MemcpyH2DKernel,
    int8_t, ops::MemcpyH2DKernel, uint8_t, ops::MemcpyH2DKernel, int,
    ops::MemcpyH2DKernel, int64_t, ops::MemcpyH2DKernel, bool,
    ops::MemcpyH2DKernel, paddle::platform::bfloat16, ops::MemcpyH2DKernel,
    paddle::platform::complex<float>, ops::MemcpyH2DKernel,
    paddle::platform::complex<double>, ops::MemcpyH2DKernel, plat::float16,
    ops::MemcpyH2DKernel, int16_t, ops::MemcpyH2DKernel);
#endif
