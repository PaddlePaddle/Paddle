// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include "paddle/fluid/operators/deformable_psroi_pooling_op.h"
#include "paddle/fluid/operators/math/blas.h"


namespace paddle {
namespace operators {
class DeformablePSROIPoolOpMaker: public framework::OpProtoAndCheckerMaker {
  public:
    void Make() override {
      AddInput("Input",
               "(Tensor), "
               "the Input of Deformable PSROIPoolOp. "
               "The format of input tensor is NCHW. Where N is batch size, "
               "C is number of input channels, "
               "H is height of the feature, and "
               "W is the width of the feature.");

      AddInput("ROIs", 
               "(LoDTensor), "
               "ROIS on feature map,"
               "The format is (N, 4),"
               "N is number of rois,"
               "4 is top left point and bottom right point on feature map, \
                eg.[batch_index, x1, y1, x2, y2].");
            
      AddInput("Trans",
               "(Tensor), "
               "offset of pixel on feature map,"
               "the format is  NCHW, "
               "N is number of ROIS, "
               "C is the distance of offset in x and y, "
               "H is pooled height, "
               "W is pooled width");

      AddAttr<int>("no_trans",
                   "(int), "
                   "whether add offset or not, "
                   "value is 0 or 1").SetDefault(0);
            
      AddAttr<float>("spatial_scale",
                     "(float), "
                     "the scale from feature map to input image").SetDefault(1.0);

      AddAttr<int>("output_dim",
                   "(int), " 
                   "number of channel of output.").SetDefault(256);

      AddAttr<int>("group_size",
                   "(int), "
                   "number of groups which the channel is divided. ").SetDefault(1);

      AddAttr<int>("pooled_size", 
                   "(int), "
                   "size after deformable psroi_pooling.").SetDefault(7);

      AddAttr<int>("part_size",
                  "(int), "
                  "the size of sharing offset on feature map.").SetDefault(7);

      AddAttr<int>("sample_per_part",
                   "(int), " 
                   "number of samples in one bin").SetDefault(4);

      AddAttr<float>("trans_std",
                     "(float), "
                     "coefficient of offset").SetDefault(0.1);

      AddOutput("Top_count",
                "(Tensor), "
                "record the number of pixel in average pooling to get one bin, "
                "the format is NCHW, "
                "N is number of batch size, "
                "C is number of channel of output, "
                "H is height of output, "
                "W is width of output.");

      AddOutput("Output",
                "(Tensor), "
                "output of deformable position sensetive roi pooling, "
                "the format is NCHW, " 
                "N is number of batch size, "
                "C is number of channel of output, "
                "H is height after pooling, "
                "W is width after pooling. ");   
       AddComment(R"DOC(
                  **Deformable ps roi pooling Operator**
                  https://arxiv.org/abs/1811.1116)DOC");
  }
};

class DeformablePSROIPoolOp : public framework::OperatorWithKernel {
  public:
    using framework::OperatorWithKernel::OperatorWithKernel;
    void InferShape(framework::InferShapeContext *ctx) const override {
      PADDLE_ENFORCE(ctx->HasInput("Input"),
                     "Input(Input) of DeformablePSROIPoolOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("ROIs"),
                     "Input(ROIs) of DeformablePSROIPoolOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("Trans"),
                     "Input(Trans) of DeformablePSROIPoolOp should not be null.");
      PADDLE_ENFORCE(ctx->HasOutput("Output"),
                     "Output(Output) of DeformablePSROIPoolOp should not be null.");
      PADDLE_ENFORCE(ctx->HasOutput("Top_count"),
                     "Output(Top_count) of DeformablePSROIPoolOp should not be null.");
      auto input_dims = ctx->GetInputDim("Input");
      auto rois_dims = ctx->GetInputDim("ROIs");  
      auto trans_dims = ctx->GetInputDim("Trans");
      PADDLE_ENFORCE(rois_dims.size() == 2,
                     "ROIs should be a 2-D LoDTensor of shape (num_rois, 4)"
                     "given as [[ x1, y1, x2, y2], ...].");
      PADDLE_ENFORCE(trans_dims.size() == 4,
                     "The format of Input tensor is (N, 2, H, W).");
              
      auto pooled_size = ctx->Attrs().Get<int>("pooled_size");         
      auto spatial_scale = ctx->Attrs().Get<float>("spatial_scale");
      auto output_dim = ctx->Attrs().Get<int>("output_dim");
      auto group_size = ctx->Attrs().Get<int>("group_size");           
      auto part_size = ctx->Attrs().Get<int>("part_size");
      auto sample_per_part = ctx->Attrs().Get<int>("sample_per_part");
      auto trans_std = ctx->Attrs().Get<float>("trans_std");
      
      PADDLE_ENFORCE(trans_std >= 0.0,
                     "trans_std must be greater than 0.0");
      PADDLE_ENFORCE_GT(pooled_size, 0,
                        "The pooled size must greater than 0");
      PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                        "The spatial scale must greater than 0");
      PADDLE_ENFORCE_GT(group_size, 0,
                      "The group_size must greater than 0"); 
      PADDLE_ENFORCE_GT(part_size, 0,
                        "The part_size must greater than 0");
      PADDLE_ENFORCE_GT(sample_per_part, 0,
                        "The sample_per_part must greater than 0");

      auto out_dims = input_dims;
      out_dims[0] = rois_dims[0];
      out_dims[1] = output_dim;
      out_dims[2] = pooled_size;
      out_dims[3] = pooled_size;

      ctx->SetOutputDim("Output", out_dims);
      ctx->SetOutputDim("Top_count", out_dims);
  }
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.device_context());
  }
};


class DeformablePSROIPoolGradOp : public framework::OperatorWithKernel {
  public:
    using framework::OperatorWithKernel::OperatorWithKernel;
      void InferShape(framework::InferShapeContext *ctx) const override {
        PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Output")),
                       "The gradient of Output should not be null.");
        PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Input")),
                       "The gradient of Input should not be null.");
        PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Trans")),
                   "The gradient of Trans should not be null.");
        ctx->SetOutputDim(framework::GradVarName("Input"), ctx->GetInputDim("Input"));
        ctx->SetOutputDim(framework::GradVarName("Trans"), ctx->GetInputDim("Trans"));
      }
  
 protected:
  framework::OpKernelType GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Trans")->type(),
                                   ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(deformable_psroi_pooling, ops::DeformablePSROIPoolOp,
                  ops::DeformablePSROIPoolOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OPERATOR(deformable_psroi_pooling_grad,
                  ops::DeformablePSROIPoolGradOp);
                  
REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling, 
    ops::DeformablePSROIPoolCPUKernel<CPU, float>,
    ops::DeformablePSROIPoolCPUKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling_grad,
    ops::DeformablePSROIPoolGradCPUKernel<CPU, float>,
    ops::DeformablePSROIPoolGradCPUKernel<CPU, double>);
