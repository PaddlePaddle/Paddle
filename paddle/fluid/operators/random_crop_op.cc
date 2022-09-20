// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/random_crop_op.h"

namespace paddle {
namespace operators {

class RandomCropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GT(
        x_dim.size(),
        static_cast<int64_t>(shape.size()),
        platform::errors::InvalidArgument(
            "The dimensions of Input(X) must be greater than the length of "
            "Attr(shape),"
            "But received dimensions of Input(X) is [%d], receivecd length"
            "of Attr(shape) is [%d].",
            x_dim.size(),
            static_cast<int64_t>(shape.size())));
    auto out_dim = phi::vectorize<int>(x_dim);
    for (size_t i = 1; i <= shape.size(); ++i) {
      size_t x_i = x_dim.size() - i;
      size_t shape_i = shape.size() - i;
      if (ctx->IsRuntime() || (x_dim[x_i] > 0 && shape[shape_i] > 0)) {
        PADDLE_ENFORCE_GE(
            x_dim[x_i],
            shape[shape_i],
            platform::errors::InvalidArgument(
                "The dimensions of Input(X) must be larger than Attr(shape),"
                "But received dimensions of Input(X) is [%d], received"
                "size of Attr(shape) is [%d].",
                x_dim[x_i],
                shape[shape_i]));
      }
      out_dim[x_i] = shape[shape_i];
    }
    ctx->SetOutputDim("Out", phi::make_ddim(out_dim));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class RandomCropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "A batch of instances to random crop.");
    AddInput("Seed", "The random seed.");
    AddOutput("Out", "The cropped instance batch.");
    AddOutput("SeedOut", "The random seed after random cropping.")
        .AsIntermediate();
    AddAttr<std::vector<int>>("shape", "The shape of a cropped instance.");
    AddAttr<int>("startup_seed",
                 "If the input 'Seed' is not initialized, the 'startup_seed' "
                 "will be used to replace it. Even so, the seed after random "
                 "crop will also be outputed to the 'SeedOut'.")
        .SetDefault(0);
    AddComment(R"DOC(
      This operator takes a batch of instance, and do random cropping on each instance.
      It means that cropping positions differs on each instance, which is determined
      by an uniform random generator. All cropped instances have the same shape, which
      is determined by the operator's attribute 'shape'.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace f = paddle::framework;
REGISTER_OPERATOR(
    random_crop,
    ops::RandomCropOp,
    ops::RandomCropOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

template <typename T>
using Kernel = ops::RandomCropKernel<phi::CPUContext, T>;
REGISTER_OP_CPU_KERNEL(random_crop,
                       Kernel<float>,
                       Kernel<int>,
                       Kernel<double>,
                       Kernel<uint8_t>,
                       Kernel<int16_t>);
