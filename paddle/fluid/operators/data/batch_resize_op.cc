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

#include "paddle/fluid/operators/data/batch_resize_op.h"

namespace paddle {
namespace operators {
namespace data {

class BatchResizeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchResize");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "BatchResize");

    auto size = ctx->Attrs().Get<std::vector<int64_t>>("size");
    PADDLE_ENFORCE_EQ(size.size(), 2,
                      platform::errors::InvalidArgument(
                          "The length of Attrs(size) should be 2."));
    PADDLE_ENFORCE_GT(size[0], 0,
                      platform::errors::InvalidArgument(
                          "h in Attr(size) of Op(BatchResize) "
                          "should be greater than 0."));
    PADDLE_ENFORCE_GT(size[1], 0,
                      platform::errors::InvalidArgument(
                          "w in Attr(size) of Op(BatchResize) "
                          "should be greater than 0."));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::UINT8, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "X") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class BatchResizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensorArray). A batch of instances to random crop.");
    AddOutput("Out", "(Tensor). The cropped instance batch.");
    AddAttr<std::vector<int64_t>>(
        "size", "expected output size of the crop, for each edge.");
    AddAttr<std::string>("interp_method",
                         "(string, default \"bilinear\"), interpolation "
                         "method, can be \"bilinear\" for "
                         "bilinear interpolation and \"nearest\" for nearest "
                         "neighbor interpolation.")
        .SetDefault("bilinear");
    AddAttr<bool>(
        "align_corners",
        "an optional bool. Defaults to True. "
        "If True, the centers of 4 corner pixels of the input and output "
        "tensors are aligned, preserving the values at the corner pixels, "
        "If False, are not aligned")
        .SetDefault(true);
    AddAttr<int>("align_mode",
                 "(int, default \'1\'), optional for bilinear interpolation, "
                 "can be \'0\' for src_idx = scale*(dst_indx+0.5)-0.5 , "
                 "can be \'1\' for src_idx = scale*dst_index .")
        .SetDefault(1);
    AddAttr<std::string>(
        "data_layout",
        "(string, default NCHW) Only used in "
        "an optional string from: \"NHWC\", \"NCHW\". "
        "Specify that the data format of the input and output data is "
        "channel_first or channel_last.")
        .SetDefault("NCHW");
    AddComment(R"DOC(
      Batch resize images
    )DOC");
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    batch_resize, ops::data::BatchResizeOp,
    ops::data::BatchResizeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(batch_resize,
                       ops::data::BatchResizeCPUKernel<uint8_t>)
