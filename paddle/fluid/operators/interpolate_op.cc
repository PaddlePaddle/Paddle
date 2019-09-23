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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using DataLayout = framework::DataLayout;

static void Interpolate2DInferShapeCheck(framework::InferShapeContext* ctx) {
  auto dim_x = ctx->GetInputDim("X");
  auto interp_method = ctx->Attrs().Get<std::string>("interp_method");

  PADDLE_ENFORCE(
      "bilinear" == interp_method || "nearest" == interp_method,
      "Interpolation method can only be \"bilinear\" or \"nearest\" when "
      "Input(X) dimension is 4");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->HasInputs("SizeTensor")) {
    // top prority size
    auto inputs_name = ctx->Inputs("SizeTensor");
    PADDLE_ENFORCE_EQ(
        inputs_name.size(), 2,
        "Input(SizeTensor)'size of Op(interpolate) must be 2. "
        "Attr(out_shape)'s length must be 2 for 4-D input tensor.");
    int out_h = ctx->Attrs().Get<int>("out_h");
    int out_w = ctx->Attrs().Get<int>("out_w");
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
    }
    ctx->SetOutputDim("Out", dim_out);

    return;
  }

  int out_h, out_w;
  if (ctx->HasInput("Scale")) {
    auto scale_tensor = ctx->GetInputDim("Scale");
    PADDLE_ENFORCE_EQ(scale_tensor.size(), 1,
                      "Scale's dimension size must be 1.");
    out_h = -1;
    out_w = -1;
  } else {
    float scale = ctx->Attrs().Get<float>("scale");
    if (scale > 0) {
      // round down
      out_h = (data_layout == DataLayout::kNCHW
                   ? static_cast<int>(dim_x[2] * scale)
                   : static_cast<int>(dim_x[1] * scale));
      out_w = (data_layout == DataLayout::kNCHW
                   ? static_cast<int>(dim_x[3] * scale)
                   : static_cast<int>(dim_x[2] * scale));
      // protect when input shape is -1
      out_h = out_h > 0 ? out_h : -1;
      out_w = out_w > 0 ? out_w : -1;
    } else {
      out_h = ctx->Attrs().Get<int>("out_h");
      out_w = ctx->Attrs().Get<int>("out_w");
    }
  }

  if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
    auto out_size_dim = ctx->GetInputDim("OutSize");
    PADDLE_ENFORCE_EQ(out_size_dim.size(), 1,
                      "OutSize's dimension size must be 1");
    PADDLE_ENFORCE_EQ(out_size_dim[0], 2, "OutSize's dim[0] must be 2");
    ctx->ShareLoD("X", "Out");
    return;
  }

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_h, out_w};
  } else {
    dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
  }
  ctx->SetOutputDim("Out", dim_out);
}

static void Interpolate3DInferShapeCheck(framework::InferShapeContext* ctx) {
  auto dim_x = ctx->GetInputDim("X");
  auto interp_method = ctx->Attrs().Get<std::string>("interp_method");

  PADDLE_ENFORCE("trilinear" == interp_method,
                 "Interpolation method can only be \"trilinear\" when Input(X) "
                 "dimension is 5");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->HasInputs("SizeTensor")) {
    // top prority size
    auto inputs_name = ctx->Inputs("SizeTensor");
    PADDLE_ENFORCE_EQ(
        inputs_name.size(), 3,
        "Input(SizeTensor)'s size of Op(interpolate) must be 3. "
        "Attr(out_shape)'s length must be 3 for 5-D input tensor.");
    int out_d = ctx->Attrs().Get<int>("out_d");
    int out_h = ctx->Attrs().Get<int>("out_h");
    int out_w = ctx->Attrs().Get<int>("out_w");
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
    }
    ctx->SetOutputDim("Out", dim_out);

    return;
  }

  int out_d, out_h, out_w;
  if (ctx->HasInput("Scale")) {
    auto scale_tensor = ctx->GetInputDim("Scale");
    PADDLE_ENFORCE_EQ(scale_tensor.size(), 1,
                      "Scale's dimension size must be 1");
    out_d = -1;
    out_h = -1;
    out_w = -1;
  } else {
    float scale = ctx->Attrs().Get<float>("scale");
    if (scale > 0) {
      // round down
      out_d = (data_layout == DataLayout::kNCHW
                   ? static_cast<int>(dim_x[2] * scale)
                   : static_cast<int>(dim_x[1] * scale));
      out_h = (data_layout == DataLayout::kNCHW
                   ? static_cast<int>(dim_x[3] * scale)
                   : static_cast<int>(dim_x[2] * scale));
      out_w = (data_layout == DataLayout::kNCHW
                   ? static_cast<int>(dim_x[4] * scale)
                   : static_cast<int>(dim_x[3] * scale));
      // protect when input shape is -1
      out_d = out_d > 0 ? out_d : -1;
      out_h = out_h > 0 ? out_h : -1;
      out_w = out_w > 0 ? out_w : -1;
    } else {
      out_d = ctx->Attrs().Get<int>("out_d");
      out_h = ctx->Attrs().Get<int>("out_h");
      out_w = ctx->Attrs().Get<int>("out_w");
    }
  }

  if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
    auto out_size_dim = ctx->GetInputDim("OutSize");
    PADDLE_ENFORCE_EQ(out_size_dim.size(), 1,
                      "OutSize's dimension size must be 1");
    PADDLE_ENFORCE_EQ(out_size_dim[0], 3, "OutSize's dim[0] must be 3");
    ctx->ShareLoD("X", "Out");
    return;
  }

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
  } else {
    dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
  }
  ctx->SetOutputDim("Out", dim_out);
}

class InterpolateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of InterpolateOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of InterpolationOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");  // NCHW format
    PADDLE_ENFORCE(dim_x.size() == 4 || dim_x.size() == 5,
                   "Input(X) dimension must be 4 or 5");

    if (dim_x.size() == 4) {
      // shape check for 2D interpolate for input tensor shape NCHW
      Interpolate2DInferShapeCheck(ctx);
    } else {  // dim_x.size() == 5
      // shape check for 3D interpolate for input tensor shape NCDHW
      Interpolate3DInferShapeCheck(ctx);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "SizeTensor" || var_name == "Scale") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class InterpolateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of interpolate operator, "
             "This is a 4-D tensor with shape of [N, C, H, W] or a "
             "5-D tensor with shape of [N, C, D, H, W].");
    AddInput("OutSize",
             "This is a 1-D tensor with two numbers to specify output size. "
             "It should be [output_height, output_width] when input is a 4-D "
             "tensor and should be [output_depth, output_height, output_width] "
             "when input is a 5-D tensor. It has a higher priority than "
             "the attr(out_d), attr(out_h), attr(out_w) and attr(scale).")
        .AsDispensable();
    AddInput("SizeTensor",
             "(vector<Tensor<int32>>, optional). If provided, interpolate will "
             "use this. The shape of the tensor in vector MUST BE [1]. "
             "It has the highest priority compare with Input(OutSize) and "
             "attr(out_d), attr(out_h), attr(out_w) and attr(scale).")
        .AsDuplicable()
        .AsDispensable();
    AddInput("Scale",
             "This is a 1-D tensor with one number to specify output scale. "
             "It has the higher priority compare with attr(scale).")
        .AsDispensable();
    AddOutput("Out",
              "The output tensor of interpolate operator, "
              "This is a tensor in same rank with Input(X).");

    AddAttr<std::string>(
        "data_layout",
        "(string, default NCHW) Only used in "
        "an optional string from: \"NHWC\", \"NCHW\". "
        "Specify that the data format of the input and output data is "
        "channel_first or channel_last.")
        .SetDefault("NCHW");
    AddAttr<int>("out_d", "output depth of interpolate op.").SetDefault(0);
    AddAttr<int>("out_h", "output height of interpolate op.").SetDefault(0);
    AddAttr<int>("out_w", "output width of interpolate op.").SetDefault(0);
    AddAttr<float>("scale", "scale factor of interpolate op.").SetDefault(0.);
    AddAttr<std::string>("interp_method",
                         "(string, default \"bilinear\"), interpolation "
                         "method, can be \"bilinear\" for "
                         "bilinear interpolation, \"trilinear\" for trilinear "
                         "interpolation and \"nearest\" for nearest "
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

          Trilinear interpolation is an extension of linear interpolation for 
          interpolating functions of three variables (e.g. D-direction, 
          H-direction and W-direction in this op) on a rectilinear 3D grid. 
          The linear interpolation is performed on three directions.

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

          Trilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
          

          For details of nearest neighbor interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

          For details of bilinear interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Bilinear_interpolation

          For details of trilinear interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Trilinear_interpolation
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
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "SizeTensor" || var_name == "Scale") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class InterpolateGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType(ForwardOp().Type() + "_grad");
    op->SetInput("X", Input("X"));
    if (ForwardOp().Inputs().count("SizeTensor") > 0) {
      op->SetInput("SizeTensor", Input("SizeTensor"));
    }
    if (ForwardOp().Inputs().count("OutSize") > 0) {
      op->SetInput("OutSize", Input("OutSize"));
    }
    if (ForwardOp().Inputs().count("Scale") > 0) {
      op->SetInput("Scale", Input("Scale"));
    }
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(InterpolateGradNoNeedBufferVarsInference,
                                      "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bilinear_interp, ops::InterpolateOp, ops::InterpolateOpMaker,
                  ops::InterpolateGradDescMaker);
REGISTER_OPERATOR(bilinear_interp_grad, ops::InterpolateOpGrad,
                  ops::InterpolateGradNoNeedBufferVarsInference);
REGISTER_OPERATOR(nearest_interp, ops::InterpolateOp, ops::InterpolateOpMaker,
                  ops::InterpolateGradDescMaker);
REGISTER_OPERATOR(nearest_interp_grad, ops::InterpolateOpGrad,
                  ops::InterpolateGradNoNeedBufferVarsInference);
REGISTER_OPERATOR(trilinear_interp, ops::InterpolateOp, ops::InterpolateOpMaker,
                  ops::InterpolateGradDescMaker);
REGISTER_OPERATOR(trilinear_interp_grad, ops::InterpolateOpGrad,
                  ops::InterpolateGradNoNeedBufferVarsInference);
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
REGISTER_OP_CPU_KERNEL(trilinear_interp, ops::InterpolateKernel<float>,
                       ops::InterpolateKernel<double>,
                       ops::InterpolateKernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(trilinear_interp_grad, ops::InterpolateGradKernel<float>,
                       ops::InterpolateGradKernel<double>);
