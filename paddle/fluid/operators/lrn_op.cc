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

#include "paddle/fluid/operators/lrn_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

template <typename T>
struct LRNFunctor<phi::CPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out,
                  phi::DenseTensor* mid,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T k,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    auto place = ctx.GetPlace();
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(ctx);
    phi::funcs::Transpose<phi::CPUContext, T, 4> transpose;
    auto& dev_ctx = ctx.template device_context<phi::CPUContext>();
    phi::DenseTensor in_transpose, mid_transpose, out_transpose;
    // if channel_last, transpose to channel_first
    if (data_layout == DataLayout::kNHWC) {
      auto in_dims = input.dims();
      std::vector<int64_t> shape(
          {in_dims[0], in_dims[3], in_dims[1], in_dims[2]});
      in_transpose.mutable_data<T>(phi::make_ddim(shape), place);
      mid_transpose.mutable_data<T>(phi::make_ddim(shape), place);
      out_transpose.mutable_data<T>(phi::make_ddim(shape), place);
      std::vector<int> axis = {0, 3, 1, 2};
      transpose(dev_ctx, input, &in_transpose, axis);
    } else {
      in_transpose = input;
      mid_transpose = *mid;
      out_transpose = *out;
      mid_transpose.mutable_data<T>(mid->dims(), place);
      out_transpose.mutable_data<T>(out->dims(), place);
    }

    const T* idata = in_transpose.data<T>();
    T* odata = out_transpose.data<T>();
    T* mdata = mid_transpose.data<T>();

    phi::DenseTensor squared;
    T* sdata = squared.mutable_data<T>({1, C + n - 1, H, W}, place);
    std::memset(sdata, 0, sizeof(T) * squared.numel());
    for (int i = 0; i < mid->numel(); ++i) {
      mdata[i] = k;
    }
    int img_size = H * W;
    int fea_size = C * img_size;
    int pre_pad = (n - 1) / 2;
    // compute batches one by one
    for (int i = 0; i < N; ++i) {
      blas.VSQUARE(fea_size, idata + i * fea_size, sdata + pre_pad * img_size);
      // init the first channel of mid
      for (int c = 0; c < n; ++c) {
        blas.AXPY(img_size, alpha, sdata + c * img_size, mdata + i * fea_size);
      }
      for (int c = 1; c < C; ++c) {
        // copy previous scale
        int mid_offset = i * fea_size + c * img_size;
        std::memcpy(mdata + mid_offset,
                    mdata + mid_offset - img_size,
                    img_size * sizeof(T));
        // add last
        blas.AXPY(img_size,
                  alpha,
                  sdata + (c + n - 1) * img_size,
                  mdata + mid_offset);
        // sub rest
        blas.AXPY(
            img_size, -alpha, sdata + (c - 1) * img_size, mdata + mid_offset);
      }
    }
    // compute the final output
    blas.VPOW(mid->numel(), mdata, -beta, odata);
    blas.VMUL(mid->numel(), odata, idata, odata);

    // if channel_last, transpose the output(NCHW) to channel_last
    if (data_layout == DataLayout::kNHWC) {
      std::vector<int> axis = {0, 2, 3, 1};
      transpose(dev_ctx, mid_transpose, mid, axis);
      transpose(dev_ctx, out_transpose, out, axis);
    }
  }
};
template struct LRNFunctor<phi::CPUContext, float>;
template struct LRNFunctor<phi::CPUContext, double>;

template <typename T>
struct LRNGradFunctor<phi::CPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& out,
                  const phi::DenseTensor& mid,
                  phi::DenseTensor* x_g,
                  const phi::DenseTensor& out_g,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    T ratio = -2 * alpha * beta;
    auto x_g_e = framework::EigenVector<T>::Flatten(*x_g);
    x_g_e = x_g_e.constant(0.0);

    auto e_x = framework::EigenTensor<T, 4>::From(x);
    auto e_x_g = framework::EigenTensor<T, 4>::From(*x_g);
    auto e_out = framework::EigenTensor<T, 4>::From(out);
    auto e_out_g = framework::EigenTensor<T, 4>::From(out_g);
    auto e_mid = framework::EigenTensor<T, 4>::From(mid);

    const int start = -(n - 1) / 2;
    const int end = start + n;
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        auto offsets = Eigen::array<int, 4>({{m, i, 0, 0}});
        auto extents = Eigen::array<int, 4>({{1, 1, H, W}});
        if (data_layout == DataLayout::kNHWC) {
          offsets = Eigen::array<int, 4>({{m, 0, 0, i}});
          extents = Eigen::array<int, 4>({{1, H, W, 1}});
        }

        auto i_x = e_x.slice(offsets, extents);
        auto i_x_g = e_x_g.slice(offsets, extents);
        auto i_out_g = e_out_g.slice(offsets, extents);
        auto i_mid = e_mid.slice(offsets, extents);

        i_x_g = i_mid.pow(-beta) * i_out_g;
        for (int c = start; c < end; c++) {
          int ch = i + c;
          if (ch < 0 || ch >= C) {
            continue;
          }

          if (data_layout != DataLayout::kNHWC) {
            offsets = Eigen::array<int, 4>({{m, ch, 0, 0}});
          } else {
            offsets = Eigen::array<int, 4>({{m, 0, 0, ch}});
          }
          auto c_out = e_out.slice(offsets, extents);
          auto c_mid = e_mid.slice(offsets, extents);
          auto c_out_g = e_out_g.slice(offsets, extents);

          i_x_g += ratio * c_out_g * c_out * i_x / c_mid;
        }
      }
    }
  }
};
template struct LRNGradFunctor<phi::CPUContext, float>;
template struct LRNGradFunctor<phi::CPUContext, double>;

class LRNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LRN");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "LRN");
    OP_INOUT_CHECK(ctx->HasOutput("MidOut"), "Output", "MidOut", "LRN");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dim.size(),
        4,
        platform::errors::InvalidArgument("Input(input) rank should be 4, "
                                          "but received input rank (%d) != 4",
                                          x_dim.size()));

    int n = ctx->Attrs().Get<int>("n");
    PADDLE_ENFORCE_GT(n,
                      0UL,
                      platform::errors::InvalidArgument(
                          "Argument(n) should be positive, "
                          "but received n(%d) not greater than 0",
                          n));
    PADDLE_ENFORCE_EQ(n % 2,
                      1UL,
                      platform::errors::InvalidArgument(
                          "Argument(n) should be odd value, "
                          "but received n(%d) is not an odd value",
                          n));

    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
    ctx->SetOutputDim("MidOut", x_dim);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
#ifdef PADDLE_WITH_MKLDNN
    if ((expected_kernel_type.data_layout_ == phi::DataLayout::ONEDNN) &&
        (tensor.layout() != phi::DataLayout::ONEDNN)) {
      auto attrs = Attrs();
      auto ar = paddle::framework::AttrReader(attrs);
      const std::string data_format = ar.Get<std::string>("data_format");
      auto dl = phi::StringToDataLayout(data_format);
      // Some models may have intentionally set "AnyLayout" for lrn
      // op. Treat this as NCHW (default data_format value)
      if (dl != phi::DataLayout::kAnyLayout) {
        return framework::OpKernelType(
            expected_kernel_type.data_type_, tensor.place(), dl);
      }
    }
#endif
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

template <typename T>
class LRNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input of LRN operator. "
             "It must be a 4D tenor with NCHW format.");
    AddOutput("Out",
              "(Tensor) The output of LRN operator, which is also the 4D "
              "tensor with NCHW format.");
    AddOutput("MidOut",
              "(Tensor) Middle result of LRN operator. It's computed in "
              "forward process and also used in backward process.");

    AddAttr<int>("n",
                 "(int default 5) "
                 "n is the \"adjacent\" kernel that maps "
                 "at the same spatial position.")
        .SetDefault(5)
        .GreaterThan(0);

    AddAttr<T>("k",
               "(float, default 2.0) "
               "k is the bias.")
        .SetDefault(2.0)
        .GreaterThan(0.0);

    AddAttr<T>("alpha",
               "(float, default 0.0001) "
               "alpha is the scale number.")
        .SetDefault(0.0001)
        .GreaterThan(0.0);

    AddAttr<T>("beta",
               "(float, default 0.75) "
               "beta is the power number.")
        .SetDefault(0.75)
        .GreaterThan(0.0);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddComment(R"DOC(
Local Response Normalization Operator.

This operator comes from the paper:
<<ImageNet Classification with Deep Convolutional Neural Networks>>.

The original formula is:

$$
Output(i, x, y) = Input(i, x, y) / \left(
k + \alpha \sum\limits^{\min(C-1, i + n/2)}_{j = \max(0, i - n/2)}
(Input(j, x, y))^2
\right)^{\beta}
$$

Function implementation:

Inputs and outputs are in NCHW or NHWC format, while input.shape.ndims() equals 4.
If NCHW, the dimensions 0 ~ 3 represent batch size, feature maps, rows,
and columns, respectively.

Input and Output in the formula above is for each map(i) of one image, and
Input(i, x, y), Output(i, x, y) represents an element in an image.

C is the number of feature maps of one image. n is a hyper-parameter
configured when operator is initialized. The sum in the denominator
is the sum of the same positions in the neighboring maps.

)DOC");
  }
};

class LRNOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LRNGrad");
    OP_INOUT_CHECK(ctx->HasInput("MidOut"), "Input", "MidOu", "LRNGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "LRNGrad");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
#ifdef PADDLE_WITH_MKLDNN
    if ((expected_kernel_type.data_layout_ == phi::DataLayout::ONEDNN) &&
        (tensor.layout() != phi::DataLayout::ONEDNN)) {
      auto attrs = Attrs();
      auto ar = paddle::framework::AttrReader(attrs);
      const std::string data_format = ar.Get<std::string>("data_format");
      auto dl = phi::StringToDataLayout(data_format);
      // Some models may have intentionally set "AnyLayout" for lrn
      // op. Treat this as NCHW (default data_format value)
      if (dl != phi::DataLayout::kAnyLayout) {
        return framework::OpKernelType(
            expected_kernel_type.data_type_, tensor.place(), dl);
      }
    }
#endif
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

template <typename T>
class LRNGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("MidOut", this->Output("MidOut"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lrn,
                  ops::LRNOp,
                  ops::LRNOpMaker<float>,
                  ops::LRNGradOpMaker<paddle::framework::OpDesc>,
                  ops::LRNGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(lrn_grad, ops::LRNOpGrad);
REGISTER_OP_CPU_KERNEL(lrn, ops::LRNKernel<phi::CPUContext, float>);
REGISTER_OP_CPU_KERNEL(lrn_grad, ops::LRNGradKernel<phi::CPUContext, float>);
