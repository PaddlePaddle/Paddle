/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/var_conv_2d_op.h"

#include <memory>
#include <vector>

#include "paddle/fluid/platform/dynload/mklml.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

void VarConv2dOpMaker::Make() {
  AddInput("X",
           "X (LoDTensor, default LoDTensor<float>) Input variable which "
           "should contain lod information.");
  AddInput("ROW", "(LoDTensor) the row variable provides lod information");
  AddInput("COLUMN",
           "(LoDTensor) the column variable provides lod information");
  AddInput("W", "W (Tensor), the filter.");
  AddAttr<int>("InputChannel", "the input filter num").SetDefault(1);
  AddAttr<int>("OutputChannel", "the output filter num").SetDefault(1);
  AddAttr<int>("StrideH", "the height of Stride").SetDefault(1);
  AddAttr<int>("StrideW", "the width of Stride").SetDefault(1);
  AddAttr<int>("KernelH", "the height of Kernel").SetDefault(1);
  AddAttr<int>("KernelW", "the width of Kernel").SetDefault(1);

  AddOutput("Out", "(LoDTensor, default LoDTensor<float>) Output variable");
  AddOutput("Col",
            "(LoDTensor, default LoDTensor<float>) the intermediate result "
            "variable");

  AddComment(R"DOC(
    Var Size Conv Operator

    This operator calculate Out = \sigma \left ( W * X + b \right ),
    only support 2-D for X.

    NOTE: only support 'float32' data type now.

  )DOC");
}

void VarConv2dOP::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("X"),
      true,
      platform::errors::NotFound("X(Input) of VarConv2dOP is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("W"),
      true,
      platform::errors::NotFound("W(Input) of VarConv2dOP is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("ROW"),
      true,
      platform::errors::NotFound("Input(ROW) of VarConv2dOP is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("COLUMN"),
      true,
      platform::errors::NotFound("Input(COLUMN) of VarConv2dOP is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("Out"),
      true,
      platform::errors::NotFound("Out(Output) of VarConv2dOP is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("Col"),
      true,
      platform::errors::NotFound("Col(Output) of VarConv2dOP is not found."));

  auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "The rank of X(Input) can't be less than 2, but received rank is %u.",
          x_dims.size()));

  auto w_dims = ctx->GetInputDim("W");

  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "Input W should be a 2-D tensor, but its actual dimension is %u.",
          w_dims.size()));
  int output_channel = ctx->Attrs().Get<int>("OutputChannel");
  int input_channel = ctx->Attrs().Get<int>("InputChannel");
  int kernel_h = ctx->Attrs().Get<int>("KernelH");
  int kernel_w = ctx->Attrs().Get<int>("KernelW");
  PADDLE_ENFORCE_EQ(
      w_dims[0],
      output_channel,
      platform::errors::InvalidArgument(
          "Input W's dimension[0] should be equal to OutputChannel, the "
          "dimension[0] is %d, OutputChannel is %d.",
          w_dims[0],
          output_channel));
  PADDLE_ENFORCE_EQ(
      w_dims[1],
      input_channel * kernel_h * kernel_w,
      platform::errors::InvalidArgument(
          "Input W's dimension[1] should be equal to InputChannel * StrideH * "
          "StrideW, the dimension[1] is %d, expected value is %d.",
          w_dims[1],
          input_channel * kernel_h * kernel_w));

  if (ctx->IsRuntime()) {
    framework::Variable* x_var =
        PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
    const auto& x_lod = x_var->Get<LoDTensor>().lod();
    PADDLE_ENFORCE_EQ(
        !x_lod.empty(),
        true,
        platform::errors::InvalidArgument("The Input(X) Tensor of VarConv2dOP "
                                          "does not contain LoD information."));

    PADDLE_ENFORCE_GE(x_lod.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The Input(X)'s lod info is corrupted."));
    PADDLE_ENFORCE_EQ(x_dims[0],
                      static_cast<int64_t>(x_lod[0].back()),
                      platform::errors::InvalidArgument(
                          "The Input(X)'s lod info mismatches the actual "
                          "tensor shape, input lod is %s, tensor shape is %s.",
                          x_lod,
                          x_dims));

    framework::Variable* row_var =
        PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("ROW")[0]);
    const auto& row_lod = row_var->Get<LoDTensor>().lod();
    PADDLE_ENFORCE_EQ(!row_lod.empty(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Input(ROW) Tensor of VarConv2dOP does not "
                          "contain LoD information."));

    framework::Variable* col_var =
        PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("COLUMN")[0]);
    const auto& col_lod = col_var->Get<LoDTensor>().lod();
    PADDLE_ENFORCE_EQ(!col_lod.empty(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Input(COLUMN) Tensor of VarConv2dOP does not "
                          "contain LoD information."));
  } else {
    std::vector<int64_t> out_dims_vec{-1};
    out_dims_vec.push_back(1);
    std::vector<int64_t> col_dims_vec{-1};
    col_dims_vec.push_back(1);
    ctx->SetOutputDim("Out", phi::make_ddim(out_dims_vec));
    ctx->SetOutputDim("Col", phi::make_ddim(col_dims_vec));
  }
}

template <typename DeviceContext, typename T>
class CPUVarConv2dOPKernel : public framework::OpKernel<T> {
 public:
  void Im2Col(const framework::ExecutionContext& ctx,
              const LoDTensor& input,
              LoDTensor* col) const {
    int input_channel = ctx.Attr<int>("InputChannel");
    auto* in_row = ctx.Input<LoDTensor>("ROW");
    auto* in_col = ctx.Input<LoDTensor>("COLUMN");
    int kernel_h = ctx.Attr<int>("KernelH");
    int kernel_w = ctx.Attr<int>("KernelW");
    int stride_h = ctx.Attr<int>("StrideH");
    int stride_w = ctx.Attr<int>("StrideW");

    int batch = input.lod()[0].size() - 1;
    const auto& bottom_offset = input.lod()[0];
    // 2-D lod info.
    const auto& offset_x = in_col->lod()[0];
    const auto& offset_y = in_row->lod()[0];

    // top offset is the whole size of each data sample
    std::vector<size_t> top_offset;
    int top_size = 0;
    top_offset.push_back(top_size);
    for (int b = 0; b < batch; ++b) {
      int width = offset_x[b + 1] - offset_x[b];
      int height = offset_y[b + 1] - offset_y[b];
      int top_im_x = 0;
      if (width == 0) {
        top_im_x = 0;
      } else {
        top_im_x = (width - 1) / stride_w + 1;
      }
      int top_im_y = 0;
      if (height == 0) {
        top_im_y = 0;
      } else {
        top_im_y = (height - 1) / stride_h + 1;
      }
      int top_x = top_im_y * top_im_x;
      int top_y = input_channel * kernel_h * kernel_w;
      top_size += top_y * top_x;
      top_offset.push_back(top_size);
    }
    framework::LoD col_lod;
    col_lod.push_back(top_offset);
    col->set_lod(col_lod);
    std::vector<int64_t> col_dims_vec{top_size};
    col_dims_vec.push_back(1);
    auto* top_data =
        col->mutable_data<T>(phi::make_ddim(col_dims_vec), ctx.GetPlace());
    auto* bottom_data = input.data<T>();

    int kernel_win_size = kernel_h * kernel_w;
    int half_kernel_h = kernel_h / 2;
    int half_kernel_w = kernel_w / 2;
    for (int b = 0; b < batch; ++b) {
      int t_offset = top_offset[b];
      int b_offset = bottom_offset[b];
      int width = offset_x[b + 1] - offset_x[b];
      int height = offset_y[b + 1] - offset_y[b];
      if (width == 0 || height == 0) {
        continue;
      }
      int top_im_x = (width - 1) / stride_w + 1;
      int top_im_y = (height - 1) / stride_h + 1;
      int top_x = top_im_y * top_im_x;
      for (int z = 0; z < input_channel; ++z) {
        int row_offset = kernel_win_size * z;
        int im_offset = z * width * height;
        for (int y = 0; y < height; y += stride_h) {
          for (int x = 0; x < width; x += stride_w) {
            int col_offset = x / stride_w + y / stride_h * top_im_x;
            for (int ky = 0; ky < kernel_h; ++ky) {
              for (int kx = 0; kx < kernel_w; ++kx) {
                int im_y = y + ky - half_kernel_h;
                int im_x = x + kx - half_kernel_w;
                if (im_x >= 0 && im_x < width && im_y >= 0 && im_y < height) {
                  top_data[t_offset +
                           (row_offset + ky * kernel_w + kx) * top_x +
                           col_offset] =
                      bottom_data[b_offset + im_offset + im_y * width + im_x];
                } else {
                  top_data[t_offset +
                           (row_offset + ky * kernel_w + kx) * top_x +
                           col_offset] = 0;
                }
              }
            }
          }
        }
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* in_row = ctx.Input<LoDTensor>("ROW");
    auto* in_col = ctx.Input<LoDTensor>("COLUMN");
    auto* w = ctx.Input<Tensor>("W");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* col = ctx.Output<LoDTensor>("Col");

    int output_channel = ctx.Attr<int>("OutputChannel");
    int input_channel = ctx.Attr<int>("InputChannel");
    int kernel_h = ctx.Attr<int>("KernelH");
    int kernel_w = ctx.Attr<int>("KernelW");
    int stride_h = ctx.Attr<int>("StrideH");
    int stride_w = ctx.Attr<int>("StrideW");

    Im2Col(ctx, *bottom, col);
    int batch = bottom->lod()[0].size() - 1;
    const auto& col_offset = col->lod()[0];
    const auto& offset_x = in_col->lod()[0];
    const auto& offset_y = in_row->lod()[0];
    std::vector<size_t> top_offset;
    int top_size = 0;
    top_offset.push_back(top_size);
    for (int b = 0; b < batch; ++b) {
      int width = offset_x[b + 1] - offset_x[b];
      int height = offset_y[b + 1] - offset_y[b];
      int top_im_x = 0;
      if (width == 0) {
        top_im_x = 0;
      } else {
        top_im_x = (width - 1) / stride_w + 1;
      }
      int top_im_y = 0;
      if (height == 0) {
        top_im_y = 0;
      } else {
        top_im_y = (height - 1) / stride_h + 1;
      }
      int top_im_size = top_im_y * top_im_x;
      top_size += output_channel * top_im_size;
      top_offset.push_back(top_size);
    }

    framework::LoD top_lod;
    top_lod.push_back(top_offset);

    top->set_lod(top_lod);
    std::vector<int64_t> top_dims_vec{top_size};
    top_dims_vec.push_back(1);
    auto* top_data =
        top->mutable_data<T>(phi::make_ddim(top_dims_vec), ctx.GetPlace());

    auto* w_data = w->data<T>();
    auto* col_data = col->data<T>();

    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(ctx);
    for (int b = 0; b < batch; ++b) {
      int top_im_size = (top_offset[b + 1] - top_offset[b]) / output_channel;
      if (top_im_size == 0) {
        continue;
      }

      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                output_channel,
                top_im_size,
                input_channel * kernel_h * kernel_w,
                1.0,
                w_data,
                col_data + col_offset[b],
                0.0,
                top_data + top_offset[b]);
    }
  }
};

template <typename T>
class VarConv2dGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("ROW", this->Input("ROW"));
    op->SetInput("COLUMN", this->Input("COLUMN"));
    op->SetInput("Col", this->Output("Col"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetAttrMap(this->Attrs());
  }
};

void VarConv2dOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                    true,
                    platform::errors::NotFound(
                        "Input(X) of SequencePadGradOp is not found."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("W"),
                    true,
                    platform::errors::NotFound(
                        "Input(W) of SequencePadGradOp is not found."));
  PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")),
                    true,
                    platform::errors::NotFound(
                        "Input(Out@GRAD) of SequencePadGradOp is not found."));

  if (ctx->HasOutput(framework::GradVarName("X"))) {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
  }
}

template <typename DeviceContext, typename T>
class CPUVarConv2dOPGradKernel : public framework::OpKernel<T> {
 public:
  void Im2ColGrad(const framework::ExecutionContext& ctx, T* top_diff) const {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* in_row = ctx.Input<LoDTensor>("ROW");
    auto* in_col = ctx.Input<LoDTensor>("COLUMN");
    auto* col = ctx.Input<LoDTensor>("Col");

    int input_channel = ctx.Attr<int>("InputChannel");
    int kernel_h = ctx.Attr<int>("KernelH");
    int kernel_w = ctx.Attr<int>("KernelW");
    int stride_h = ctx.Attr<int>("StrideH");
    int stride_w = ctx.Attr<int>("StrideW");

    auto* dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    memset(dx_data, 0.0, x->dims()[0] * x->dims()[1] * sizeof(T));

    const auto& bottom_offset = x->lod()[0];
    const auto& offset_x = in_col->lod()[0];
    const auto& offset_y = in_row->lod()[0];
    const auto& top_offset = col->lod()[0];
    int batch = x->lod()[0].size() - 1;
    int kernel_win_size = kernel_h * kernel_w;
    int half_kernel_h = kernel_h / 2;
    int half_kernel_w = kernel_w / 2;
    for (int b = 0; b < batch; ++b) {
      int t_offset = top_offset[b];
      int b_offset = bottom_offset[b];
      int width = offset_x[b + 1] - offset_x[b];
      int height = offset_y[b + 1] - offset_y[b];
      if (width == 0 || height == 0) {
        continue;
      }
      int top_im_x = (width - 1) / stride_w + 1;
      int top_im_y = (height - 1) / stride_h + 1;
      int top_x = top_im_y * top_im_x;
      for (int z = 0; z < input_channel; ++z) {
        int row_offset = kernel_win_size * z;
        int im_offset = z * width * height;
        for (int y = 0; y < height; y += stride_h) {
          for (int x = 0; x < width; x += stride_w) {
            int col_offset = x / stride_w + y / stride_h * top_im_x;
            for (int ky = 0; ky < kernel_h; ++ky) {
              for (int kx = 0; kx < kernel_w; ++kx) {
                int im_y = y + ky - half_kernel_h;
                int im_x = x + kx - half_kernel_w;
                if (im_x >= 0 && im_x < width && im_y >= 0 && im_y < height) {
                  dx_data[b_offset + im_offset + im_y * width + im_x] +=
                      top_diff[t_offset +
                               (row_offset + ky * kernel_w + kx) * top_x +
                               col_offset];
                }
              }
            }
          }
        }
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    auto* col = ctx.Input<LoDTensor>("Col");
    auto* out = ctx.Input<LoDTensor>("Out");

    int output_channel = ctx.Attr<int>("OutputChannel");
    int input_channel = ctx.Attr<int>("InputChannel");
    int kernel_h = ctx.Attr<int>("KernelH");
    int kernel_w = ctx.Attr<int>("KernelW");

    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_w = ctx.Output<Tensor>(framework::GradVarName("W"));

    Tensor col_grad;
    col_grad.Resize(col->dims());
    auto* col_diff = col_grad.mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* w_diff = d_w->mutable_data<T>(ctx.GetPlace());

    memset(dx_data, 0.0, x->dims()[0] * x->dims()[1] * sizeof(T));
    memset(w_diff, 0.0, w->dims()[0] * w->dims()[1] * sizeof(T));
    memset(col_diff, 0.0, col->dims()[0] * col->dims()[1] * sizeof(T));
    auto* top_diff = d_out->data<T>();
    auto* w_data = w->data<T>();
    auto* col_data = col->data<T>();
    int batch = x->lod()[0].size() - 1;
    const auto& top_offset = out->lod()[0];
    const auto& col_offset = col->lod()[0];
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(ctx);
    for (int b = 0; b < batch; ++b) {
      int top_im_size = (top_offset[b + 1] - top_offset[b]) / output_channel;
      if (top_im_size == 0) {
        continue;
      }

      blas.GEMM(CblasTrans,
                CblasNoTrans,
                input_channel * kernel_h * kernel_w,
                top_im_size,
                output_channel,
                1.0,
                w_data,
                top_diff + top_offset[b],
                1.0,
                col_diff + col_offset[b]);

      blas.GEMM(CblasNoTrans,
                CblasTrans,
                output_channel,
                input_channel * kernel_h * kernel_w,
                top_im_size,
                1.0,
                top_diff + top_offset[b],
                col_data + col_offset[b],
                1.0,
                w_diff);
    }
    Im2ColGrad(ctx, col_diff);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(var_conv_2d,
                  ops::VarConv2dOP,
                  ops::VarConv2dOpMaker,
                  ops::VarConv2dGradMaker<paddle::framework::OpDesc>,
                  ops::VarConv2dGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(var_conv_2d_grad, ops::VarConv2dOpGrad);

REGISTER_OP_CPU_KERNEL(var_conv_2d,
                       ops::CPUVarConv2dOPKernel<phi::CPUContext, float>);
//     ops::CPUVarConv2dOPKernel<phi::CPUContext,
//                                       double>
REGISTER_OP_CPU_KERNEL(var_conv_2d_grad,
                       ops::CPUVarConv2dOPGradKernel<phi::CPUContext, float>);
//     ops::CPUVarConv2dOPGradKernel<phi::CPUContext,
//                                           double>
