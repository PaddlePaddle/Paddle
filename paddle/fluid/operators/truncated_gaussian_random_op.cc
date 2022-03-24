/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <limits>
#include <random>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/truncated_gaussian_random_op.h"
#include "paddle/phi/infermeta/nullary.h"

namespace paddle {
namespace operators {

class TruncatedGaussianRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout{framework::DataLayout::kAnyLayout};
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.device_context(), layout, library);
  }
};

class TruncatedGaussianRandomOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Output tensor of truncated gaussian random op.");

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
    AddComment(R"DOC(
TruncatedGaussianRandom Operator.

Used to initialize tensors with truncated gaussian random generator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(
    truncated_gaussian_random, TruncatedGaussianRandomInferShapeFunctor,
    PD_INFER_META(phi::TruncatedGaussianRandomInferMeta));

REGISTER_OPERATOR(
    truncated_gaussian_random, ops::TruncatedGaussianRandomOp,
    ops::TruncatedGaussianRandomOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    TruncatedGaussianRandomInferShapeFunctor);
