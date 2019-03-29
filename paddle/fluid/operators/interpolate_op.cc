/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/interpolate_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class InterpolateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of InterpolateOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of InterpolationOp should not be null.");

    auto interp_method = ctx->Attrs().Get<std::string>("interp_method");
    PADDLE_ENFORCE(
        "bilinear" == interp_method || "nearest" == interp_method,
        "Interpolation method can only be \"bilinear\" or \"nearest\".");

    auto dim_x = ctx->GetInputDim("X");  // NCHW format
    PADDLE_ENFORCE_EQ(dim_x.size(), 4, "X's dimension must be 4");

    int out_h, out_w;
    float scale = ctx->Attrs().Get<float>("scale");
    if (scale > 0) {
      // round down
      out_h = static_cast<int>(dim_x[2] * scale);
      out_w = static_cast<int>(dim_x[3] * scale);
    } else {
      out_h = ctx->Attrs().Get<int>("out_h");
      out_w = ctx->Attrs().Get<int>("out_w");
    }

    if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
      auto out_size_dim = ctx->GetInputDim("OutSize");
      PADDLE_ENFORCE_EQ(out_size_dim.size(), 1,
                        "OutSize's dimension size must be 1");
      PADDLE_ENFORCE_EQ(out_size_dim[0], 2, "OutSize's dim[0] must be 2");
      ctx->ShareLoD("X", "Out");
      return;
    }
    std::vector<int64_t> dim_out({dim_x[0], dim_x[1], out_h, out_w});
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class InterpolateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of interpolate operator, "
             "This is a 4-D tensor with shape of [N,  C, H, w].");
    AddInput("OutSize",
             "This is a 1-D tensor with two numbers to specify output size. "
             "The first number is height and the second number is width.")
        .AsDispensable();
    AddOutput("Out",
              "The output tensor of interpolate operator, "
              "This is a 4-D tensor with shape of [N, C, H, W].");

    AddAttr<int>("out_h", "output height of interpolate op.");
    AddAttr<int>("out_w", "output width of interpolate op.");
    AddAttr<float>("scale", "scale factor of interpolate op.").SetDefault(0.);
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
    AddComment(R"DOC(
          This operator samples input X to given output shape by using specified
          interpolation method, the interpolation methods can be \"nearest\"
          for nearest neighbor interpolation and \"bilinear\" for bilinear 
          interpolation.

          Nearest neighbor interpolation is to perform nearest neighbor interpolation
          in both the 3rd dimention(in height direction) and the 4th dimention(in width 
          direction) on input tensor.
            
          Bilinear interpolation is an extension of linear interpolation for 
          interpolating functions of two variables (e.g. H-direction and 
          W-direction in this op) on a rectilinear 2D grid. The key idea is 
          to perform linear interpolation first in one direction, and then 
          again in the other direction.

          Align_corners and align_mode are optinal parameters,the calculation method 
          of interpolation can be selected by them.
          
          Example:

          For scale:
          
            if align_corners = True and out_{size}>1 :

              scale_{factor} = (in_{size}-1.0)/(out_{size}-1.0)
            
            else:
              
              scale_{factor} = float(in_{size}/out_{size})
            
          
          Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = \left \lfloor {H_{in} * scale_{}factor}} \right \rfloor
              W_out = \left \lfloor {W_{in} * scale_{}factor}} \right \rfloor

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

          Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

          

          For details of nearest neighbor interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

          For details of bilinear interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Bilinear_interpolation
         )DOC");
  }
};

class InterpolateOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bilinear_interp, ops::InterpolateOp, ops::InterpolateOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(bilinear_interp_grad, ops::InterpolateOpGrad);
REGISTER_OPERATOR(nearest_interp, ops::InterpolateOp, ops::InterpolateOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(nearest_interp_grad, ops::InterpolateOpGrad);
REGISTER_OP_CPU_KERNEL(bilinear_interp, ops::InterpolateKernel<float>,
                       ops::InterpolateKernel<double>,
                       ops::InterpolateKernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(bilinear_interp_grad, ops::InterpolateGradKernel<float>,
                       ops::InterpolateGradKernel<double>);
REGISTER_OP_CPU_KERNEL(nearest_interp, ops::InterpolateKernel<float>,
                       ops::InterpolateKernel<double>,
                       ops::InterpolateKernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(nearest_interp_grad, ops::InterpolateGradKernel<float>,
                       ops::InterpolateGradKernel<double>);
