/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class PolygonBoxTransformCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()),
        true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    auto* in = ctx.Input<phi::DenseTensor>("Input");
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    auto* out = ctx.Output<phi::DenseTensor>("Output");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = in_dims[0];
    int geo_channel = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int id = 0;
    for (int id_n = 0; id_n < batch_size * geo_channel; ++id_n) {
      for (int id_h = 0; id_h < height; ++id_h) {
        for (int id_w = 0; id_w < width; ++id_w) {
          id = id_n * height * width + width * id_h + id_w;
          if (id_n % 2 == 0) {
            out_data[id] = id_w * 4 - in_data[id];
          } else {
            out_data[id] = id_h * 4 - in_data[id];
          }
        }
      }
    }
  }
};

class PolygonBoxTransformOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Input"), "Input", "Input", "polygon_box_transform");
    OP_INOUT_CHECK(
        ctx->HasOutput("Output"), "Output", "Output", "polygon_box_transform");

    auto in_dim = ctx->GetInputDim("Input");

    PADDLE_ENFORCE_EQ(
        in_dim.size(),
        4,
        platform::errors::InvalidArgument(
            "input's rank must be 4. But received: Input rank is [%d]",
            in_dim.size()));
    PADDLE_ENFORCE_EQ(in_dim[1] % 2,
                      0,
                      platform::errors::InvalidArgument(
                          "input's second dimension must be even. But "
                          "received: Input 2nd dimension is [%d]",
                          in_dim[1]));

    ctx->SetOutputDim("Output", in_dim);
  }
};

class PolygonBoxTransformOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "The input with shape [batch_size, geometry_channels, height, width]");
    AddOutput("Output", "The output with the same shape as input");

    AddComment(R"DOC(
PolygonBoxTransform Operator.

PolygonBoxTransform Operator is used to transform the coordinate shift to the real coordinate.

The input is the final geometry output in detection network.
We use 2*n numbers to denote the coordinate shift from n corner vertices of
the polygon_box to the pixel location. As each distance offset contains two numbers (xi, yi),
the geometry output contains 2*n channels.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    polygon_box_transform,
    ops::PolygonBoxTransformOp,
    ops::PolygonBoxTransformOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    polygon_box_transform,
    ops::PolygonBoxTransformCPUKernel<paddle::platform::CPUPlace, float>,
    ops::PolygonBoxTransformCPUKernel<paddle::platform::CPUPlace, double>);
