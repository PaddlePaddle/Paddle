// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "thrust/device_vector.h"
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

class FFTC2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // TODO(chenfeiyu): check shape and dim here and generate output dim
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // TODO(chenfeiyu): get output dtype from X
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    // TODO(chenfeiyu): get kernel type
  }
};

class FFTC2COpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_c2c op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_c2c op.");
    AddAttr<std::vector<int64_t>>("s", "std::vector<int64_t>, the fft shape.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<int64_t>("norm", "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddComment(R"DOC(
      // add doc here
    )DOC");
  }
};

class FFTC2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShaoe(framework::InferShapeContext* ctx) const override {
    // TODO(chenfeiyu): check shape and dim here and generate output dim
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // TODO(chenfeiyu): get output dtype from DOut
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    // TODO(chenfeiyu): get kernel type
  }
};

template <typename T>
class FFTC2CGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_c2c_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fft_c2c, ops::FFTC2COp, ops::FFTC2COpMaker,
                  ops::FFTC2CGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTC2CGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2c_grad, ops::FFTC2CGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, double>);
