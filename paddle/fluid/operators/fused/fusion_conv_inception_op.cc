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

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"

namespace paddle {
namespace operators {

class ConvInceptionFusionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    // 1 x
    auto in_dims = ctx->GetInputDim("Input");
    // 4 filters
    auto w_dims = ctx->GetInputsDim("Filter");

    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        4,
        platform::errors::InvalidArgument("Conv intput should be 4-D tensor."));
    PADDLE_ENFORCE_EQ(
        w_dims.size(),
        4,
        platform::errors::InvalidArgument("There should be 4 filters."));
    PADDLE_ENFORCE_EQ(w_dims[0][1],
                      in_dims[1],
                      platform::errors::InvalidArgument(
                          "Invalid fileter channel number %d, which should be "
                          "equal to input channel number %d.",
                          w_dims[0][1],
                          in_dims[1]));
    PADDLE_ENFORCE_EQ(w_dims[1][1],
                      in_dims[1],
                      platform::errors::InvalidArgument(
                          "Invalid fileter channel number %d, which should be "
                          "equal to input channel number %d.",
                          w_dims[1][1],
                          in_dims[1]));

    int n = in_dims[0];
    // compute output channel
    // 1st channel
    int c = w_dims[0][0];
    // add 2nd channel
    c += (w_dims[1][0] - w_dims[2][1] * 2);
    // add 3rd channel
    c += (w_dims[2][0] - w_dims[3][1]);
    // add 4-th channel
    c += w_dims[3][0];

    int h = in_dims[2];
    int w = in_dims[3];

    ctx->SetOutputDim("Output", {n, c, h, w});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class ConvInceptionFusionOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  void Make() override {
    AddInput("Input", "(Tensor) NCHW layout.");
    AddInput("Filter", "(vector<Tensor>) 4 aggregated filters").AsDuplicable();
    AddInput("Bias", "(vector<Tensor>) it's length is equal to Filter")
        .AsDuplicable();
    AddOutput("Output",
              "(Tensor) The output tensor of convolution operator. "
              "The format of output tensor is also NCHW.");
    AddOutput("TempOutput", "").AsDuplicable();
    AddAttr<std::string>(
        "pooling_type",
        "(string), pooling type, can be \"max\" for max-pooling "
        "and \"avg\" for average-pooling.")
        .InEnum({"max", "avg"});
    AddAttr<bool>(
        "exclusive",
        "(bool, default True) When true, will exclude the zero-padding in the "
        "averaging calculating, otherwise, include the zero-padding. Note, it "
        "is only used when pooling_type is avg. The default is True.")
        .SetDefault(true);
    AddAttr<std::string>(
        "activation",
        "The activation type can be 'identity', 'sigmoid', 'relu', 'relu6' "
        "'relux' , 'tanh', 'band_pass'")
        .SetDefault("relu");
    AddAttr<int>("workspace_size_MB",
                 "Only used in cudnn kernel. Need set use_cudnn to true."
                 "workspace size for cudnn, in MB, "
                 "workspace is a section of GPU memory which will be "
                 "allocated/freed each time the operator runs, larger "
                 "workspace size can increase performance but also requires "
                 "better hardware. This size should be chosen carefully.")
        .SetDefault(phi::backends::gpu::GetDefaultConvWorkspaceSizeLimitMB());
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    conv2d_inception_fusion,
    ops::ConvInceptionFusionOp,
    ops::ConvInceptionFusionOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
