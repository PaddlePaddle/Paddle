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

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct DirichletSampler<platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, const Tensor* alpha,
                  Tensor* out) {
    auto& dev_ctx = ctx.device_context<platform::CPUDeviceContext>();
    std::random_device rd;
    std::mt19937 generator(rd());

    auto uniform = [&generator]() -> T {
      std::uniform_real_distribution<T> u(0.0, 1.0);
      return u(generator);
    };
    BaseSampler<T, decltype(uniform)> standard_uniform(uniform);

    auto normal = [&generator]() {
      std::normal_distribution<T> n(0.0, 1.0);
      return n(generator);
    };
    BaseSampler<T, decltype(normal)> standard_normal(normal);

    framework::Tensor gamma;
    gamma.mutable_data<T>(alpha->dims(), dev_ctx.GetPlace());

    GammaSampler<T, decltype(uniform), decltype(normal)> gamma_sampler(
        alpha->data<T>(), gamma.data<T>(), standard_uniform, standard_normal);

    platform::ForRange<platform::CPUDeviceContext> for_range(dev_ctx,
                                                             alpha->numel());
    for_range(gamma_sampler);

    framework::Tensor gamma_sum;
    auto gamma_sum_dims_vector = vectorize(gamma.dims());
    gamma_sum_dims_vector[gamma_sum_dims_vector.size() - 1] = 1;
    gamma_sum.mutable_data<T>(framework::make_ddim(gamma_sum_dims_vector),
                              dev_ctx.GetPlace());

    std::vector<int> reduce_dims;
    reduce_dims.push_back(gamma_sum_dims_vector.size() - 1);
    ReduceKernelFunctor<platform::CPUDeviceContext, T, SumFunctor>(
        &gamma, &gamma_sum, reduce_dims, true, false, ctx)
        .template apply<T>();

    ElementwiseComputeEx<DivFunctor<T>, platform::CPUDeviceContext, T, T>(
        ctx, &gamma, &gamma_sum, -1, DivFunctor<T>(), out);
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
