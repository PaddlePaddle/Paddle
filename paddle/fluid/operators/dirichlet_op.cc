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

#include "paddle/fluid/operators/dirichlet_op.h"

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

namespace paddle {
namespace operators {
template <typename T, typename UniformSamplerT, typename NormalSamplerT>
struct GammaCPUFunctor {
  GammaCPUFunctor(const T* alpha, T* gamma,
                  BaseSampler<T, UniformSamplerT> uniform,
                  BaseSampler<T, NormalSamplerT> normal)
      : alpha_(alpha), gamma_(gamma), uniform_(uniform), normal_(normal) {}

  HOST void operator()(int64_t index) {
    auto sample = sample_gamma<T, T, UniformSamplerT, NormalSamplerT>(
        alpha_[index], uniform_, normal_);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  BaseSampler<T, UniformSamplerT> uniform_;
  BaseSampler<T, NormalSamplerT> normal_;
};

template <typename T>
struct DirichletSampler<platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, const Tensor* alpha,
                  Tensor* out) {
    auto& dev_ctx = ctx.device_context<platform::CPUDeviceContext>();

    auto p_gen = framework::DefaultCPUGenerator();
    auto generator = p_gen->GetCPUEngine();

    auto uniform = [&generator]() -> T {
      std::uniform_real_distribution<T> u(0.0, 1.0);
      return u(*generator);
    };
    BaseSampler<T, decltype(uniform)> standard_uniform(uniform);

    auto normal = [&generator]() {
      std::normal_distribution<T> n(0.0, 1.0);
      return n(*generator);
    };
    BaseSampler<T, decltype(normal)> standard_normal(normal);

    // sample from K gamma distributions, where K=alpha.numel()
    framework::Tensor gamma_samples;
    gamma_samples.mutable_data<T>(alpha->dims(), dev_ctx.GetPlace());
    GammaCPUFunctor<T, decltype(uniform), decltype(normal)> gamma_functor(
        alpha->data<T>(), gamma_samples.data<T>(), standard_uniform,
        standard_normal);
    platform::ForRange<platform::CPUDeviceContext> for_range(dev_ctx,
                                                             alpha->numel());
    for_range(gamma_functor);

    // normalize them into a simplex, along the last axis
    framework::Tensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.mutable_data<T>(new_shape, dev_ctx.GetPlace());

    ReduceKernelFunctor<platform::CPUDeviceContext, T, SumFunctor>(
        &gamma_samples, &gamma_sum, {new_shape.size() - 1}, true, false, ctx)
        .template apply<T>();
    ElementwiseComputeEx<DivFunctor<T>, platform::CPUDeviceContext, T, T>(
        ctx, &gamma_samples, &gamma_sum, -1, DivFunctor<T>(), out);
  }
};

class DirichletOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Alpha", "(Tensor), The dirichlet Alpha parameter");
    AddOutput("Out", "(Tensor), The output tensor of sample");
    AddComment(R"DOC(Sample random data from dirichlet distribution.)DOC");
  }
};

class DirichletOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Alpha"), "Input", "Alpha", "dirichlet");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "dirichlet");
    const auto alpha_dim = ctx->GetInputDim("Alpha");
    PADDLE_ENFORCE_GE(alpha_dim.size(), 1,
                      platform::errors::InvalidArgument(
                          "ShapeError: The number of dimensions of 'Alpha' "
                          "must be greater than or euqal to 1. "
                          "But received Alpha's dimensions = %d,",
                          alpha_dim.size()));
    ctx->ShareDim("Alpha", /*->*/ "Out");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(dirichlet, paddle::operators::DirichletOp,
                             paddle::operators::DirichletOpMaker);
REGISTER_OP_CPU_KERNEL(
    dirichlet,
    paddle::operators::DirichletKernel<paddle::platform::CPUDeviceContext,
                                       float>,
    paddle::operators::DirichletKernel<paddle::platform::CPUDeviceContext,
                                       double>);
