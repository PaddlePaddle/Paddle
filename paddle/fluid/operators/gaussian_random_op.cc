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
#include "paddle/fluid/operators/gaussian_random_op.h"
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
using CPUGaussianRandomKernel =
    GaussianRandomKernel<platform::CPUDeviceContext, T>;

class GaussianRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GaussianRandomOp should not be null.");
    auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
    std::vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto dim : shape) {
      temp.push_back(static_cast<int64_t>(dim));
    }
    PADDLE_ENFORCE(shape.size() > 0UL,
                   "shape can be one int or array. shape must be set.");
    ctx->SetOutputDim("Out", framework::make_ddim(temp));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout{framework::DataLayout::kAnyLayout};

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.device_context(), layout, library);
  }
};

class GaussianRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Output matrix of gaussian random op");

    AddAttr<std::vector<int>>("shape",
                              "(vector<int>) "
                              "The dimension of random tensor.");
    AddAttr<float>("mean",
                   "(float, default 0.0) "
                   "mean of random tensor.")
        .SetDefault(.0f);
    AddAttr<float>("std",
                   "(float, default 1.0) "
                   "std of random tensor.")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "(int, default 0) "
                 "Random seed of generator."
                 "0 means use system wide seed."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time.")
        .SetDefault(0);
    AddAttr<int>("dtype",
                 "(int, default 5(FP32)) "
                 "Output data type.")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
GaussianRandom Operator.

Used to initialize tensors with gaussian random generator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(gaussian_random, ops::GaussianRandomOp,
                             ops::GaussianRandomOpMaker);
REGISTER_OP_CPU_KERNEL(gaussian_random, ops::CPUGaussianRandomKernel<float>,
                       ops::CPUGaussianRandomKernel<double>);
