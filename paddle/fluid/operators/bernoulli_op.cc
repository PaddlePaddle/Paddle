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
#include "paddle/fluid/operators/bernoulli_op.h"

#include <algorithm>
#include <string>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"

namespace paddle {
namespace operators {

class BernoulliOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A tensor with probabilities for generating the random binary "
             "number");
    AddOutput("Out", "A Tensor filled with random binary number");
    AddComment(R"DOC(
This OP returns a Tensor filled with random binary(0 or 1) number from a Bernoulli distribution.

    Out ~ Bernoulli(X)

)DOC");
  }
};

class BernoulliOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return UnaryOpUnchangedInferShape(ctx);
  }
};

// It seems that Eigen::Tensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T>
class BernoulliOpKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto *in_data = x->data<T>();
    auto *out_data = out->mutable_data<T>(ctx.GetPlace());

    int64_t size = x->numel();
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    auto gen_ptr = framework::Generator::GetInstance();
    std::mt19937_64 &gen_engine = gen_ptr->GetCPUEngine();

    for (int64_t i = 0; i < size; ++i) {
      out_data[i] = BernoulliFunctor(in_data[i], dist(gen_engine));
    }
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    bernoulli, ops::BernoulliOp, ops::BernoulliOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(bernoulli,
                       ops::BernoulliOpKernel<plat::CPUDeviceContext, float>,
                       ops::BernoulliOpKernel<plat::CPUDeviceContext, double>);
