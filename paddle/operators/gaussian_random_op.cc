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

#include <random>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class CPUGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::normal_distribution<T> dist(mean, std);
    int64_t size = tensor->numel();
    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
  }
};

class GaussianRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GaussianRandomOp should not be null.");
    auto dims = ctx->Attrs().Get<std::vector<int>>("dims");
    std::vector<int64_t> temp;
    temp.reserve(dims.size());
    for (auto dim : dims) {
      temp.push_back(static_cast<int64_t>(dim));
    }
    PADDLE_ENFORCE(dims.size() > 0UL,
                   "dims can be one int or array. dims must be set.");
    ctx->SetOutputDim("Out", framework::make_ddim(temp));
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return static_cast<framework::DataType>(Attr<int>("data_type"));
  }
};

class GaussianRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GaussianRandomOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "output matrix of random op");
    AddComment(R"DOC(
GaussianRandom operator.
Use to initialize tensor with gaussian random generator.
)DOC");

    AddAttr<std::vector<int>>("dims", "The dimension of random tensor.");
    AddAttr<float>("mean", "mean of random tensor.").SetDefault(.0f);
    AddAttr<float>("std", "std of random tensor.").SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed of generator."
                 "0 means use system wide seed")
        .SetDefault(0);
    AddAttr<int>("data_type", "output data type")
        .SetDefault(framework::DataType::FP32);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(gaussian_random, ops::GaussianRandomOp,
                             ops::GaussianRandomOpMaker);
REGISTER_OP_CPU_KERNEL(gaussian_random, ops::CPUGaussianRandomKernel<float>);
