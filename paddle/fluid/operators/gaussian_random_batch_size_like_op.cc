/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/batch_size_like.h"

namespace paddle {
namespace operators {

class GaussianRandomBatchSizeLikeOp : public BatchSizeLikeOp {
 protected:
  using BatchSizeLikeOp::BatchSizeLikeOp;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class GaussianRandomBatchSizeLikeOpMaker : public BatchSizeLikeOpMaker {
 protected:
  void Apply() override {
    AddAttr<float>("mean",
                   "(float, default 0.0) "
                   "The mean (or center) of the gaussian distribution.")
        .SetDefault(.0f);
    AddAttr<float>("std",
                   "(float, default 1.0) "
                   "The standard deviation (std, or spread) of the "
                   "gaussian distribution.")
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

Used to initialize tensors with gaussian random generator.
The defalut mean of the distribution is 0. and defalut standard
deviation (std) of the distribution is 1.. Uers can set mean and std
by input arguments.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(gaussian_random_batch_size_like,
                  paddle::operators::GaussianRandomBatchSizeLikeOp,
                  paddle::operators::GaussianRandomBatchSizeLikeOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::BatchSizeLikeNoNeedBufferVarsInference);

// Kernels are registered in gaussian_random_op.cc and gaussian_random_op.cu
