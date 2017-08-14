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
class GaussianRandomKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.op_.GetAttr<float>("mean");
    float std = context.op_.GetAttr<float>("std");
    auto* tensor = context.Output<framework::Tensor>(0);
    T* data = tensor->mutable_data<T>(context.GetPlace());

    // TODO(dzh): attribute does not support unsigned int.
    // And we need a global random seed configuration.
    int seed = context.op_.GetAttr<int>("seed");
    if (seed == 0) {
      seed = std::random_device()();
    }
    std::mt19937 g(seed);
    std::normal_distribution<T> distribution(mean, std);
    ssize_t size = framework::product(tensor->dims());
    for (int i = 0; i < size; ++i) {
      data[i] = distribution(g);
    }
  }
};

class GaussianRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>(0);
    auto dims = GetAttr<std::vector<int>>("dims");
    PADDLE_ENFORCE(dims.size() > 0UL,
                   "dims can be one int or array. dims must be set.");
    tensor->Resize(framework::make_ddim(dims));
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
    AddAttr<float>("mean", "mean value of random.").SetDefault(.0f);
    AddAttr<float>("std", "minimum value of random value.").SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed of generator."
                 "0 means use system wide seed")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(gaussian_random, ops::GaussianRandomOp,
                             ops::GaussianRandomOpMaker);
REGISTER_OP_CPU_KERNEL(gaussian_random, ops::GaussianRandomKernel<float>);
