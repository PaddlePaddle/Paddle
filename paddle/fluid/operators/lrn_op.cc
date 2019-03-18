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
#include <string>
#include "paddle/fluid/operators/math/blas.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
struct LRNFunctor<platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor& input, framework::Tensor* out,
                  framework::Tensor* mid, int N, int C, int H, int W, int n,
                  T k, T alpha, T beta) {
    const T* idata = input.data<T>();
    auto place = ctx.GetPlace();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    T* odata = out->mutable_data<T>(place);
    T* mdata = mid->mutable_data<T>(place);
    Tensor squared;
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
        std::memcpy(mdata + mid_offset, mdata + mid_offset - img_size,
                    img_size * sizeof(T));
        // add last
        blas.AXPY(img_size, alpha, sdata + (c + n - 1) * img_size,
                  mdata + mid_offset);
        // sub rest
        blas.AXPY(img_size, -alpha, sdata + (c - 1) * img_size,
                  mdata + mid_offset);
      }
    }
    // compute the final output
    blas.VPOW(mid->numel(), mdata, -beta, odata);
    blas.VMUL(mid->numel(), odata, idata, odata);
  }
};
template struct LRNFunctor<platform::CPUDeviceContext, float>;
template struct LRNFunctor<platform::CPUDeviceContext, double>;

template <typename T>
struct LRNGradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor& x, const framework::Tensor& out,
                  const framework::Tensor& mid, framework::Tensor* x_g,
                  const framework::Tensor& out_g, int N, int C, int H, int W,
                  int n, T alpha, T beta) {
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
        auto i_x = e_x.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                             Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_x_g = e_x_g.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_out_g = e_out_g.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                     Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_mid = e_mid.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

        i_x_g = i_mid.pow(-beta) * i_out_g;
        for (int c = start; c < end; c++) {
          int ch = i + c;
          if (ch < 0 || ch >= C) {
            continue;
          }

          auto c_out = e_out.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                   Eigen::array<int, 4>({{1, 1, H, W}}));

          auto c_mid = e_mid.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                   Eigen::array<int, 4>({{1, 1, H, W}}));

          auto c_out_g = e_out_g.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                       Eigen::array<int, 4>({{1, 1, H, W}}));

          i_x_g += ratio * c_out_g * c_out * i_x / c_mid;
        }
      }
    }
  }
};
template struct LRNGradFunctor<platform::CPUDeviceContext, float>;
template struct LRNGradFunctor<platform::CPUDeviceContext, double>;

namespace {
framework::OpKernelType GetExpectedLRNKernel(
    const framework::ExecutionContext& ctx) {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = ctx.Attr<std::string>("data_format");
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(ctx.Input<Tensor>("X")->type(), ctx.GetPlace(),
                                 layout_, library_);
}
}  // namespace

class LRNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of LRNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LRNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MidOut"),
                   "MidOut(Out) of LRNOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), 4, "Input(X)'rank of LRNOp should be 4.");

    int n = ctx->Attrs().Get<int>("n");
    PADDLE_ENFORCE(n > 0 && n % 2 == 1, "n should be positive odd value");

    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
    ctx->SetOutputDim("MidOut", x_dim);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetExpectedLRNKernel(ctx);
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
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);

    AddComment(R"DOC(
Local Response Normalization Operator.

This operator comes from the paper:
<<ImageNet Classification with Deep Convolutional Neural Networks>>.

The original formula is:

$$
Output(i, x, y) = Input(i, x, y) / \left(
k + \alpha \sum\limits^{\min(C, c + n/2)}_{j = \max(0, c - n/2)}
(Input(j, x, y))^2
\right)^{\beta}
$$

Function implementation:

Inputs and outpus are in NCHW format, while input.shape.ndims() equals 4.
And dimensions 0 ~ 3 represent batch size, feature maps, rows,
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
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("MidOut"), "Input(MidOut) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetExpectedLRNKernel(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lrn, ops::LRNOp, ops::LRNOpMaker<float>,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(lrn_grad, ops::LRNOpGrad);
REGISTER_OP_CPU_KERNEL(
    lrn, ops::LRNKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    lrn_grad, ops::LRNGradKernel<paddle::platform::CPUDeviceContext, float>);
