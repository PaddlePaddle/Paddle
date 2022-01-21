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

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#if !defined(PADDLE_WITH_CUDA) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
#include "paddle/fluid/operators/jit/kernels.h"
#endif
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace platform {
class CPUDeviceContext;
class CUDADeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

// Wrap RowwiseMean and ColwiseMean.
// Reuse the cpu codes and replace the gpu codes with cublas_gemv, which is
// significantly faster. Unlike the RowwiseMean and ColwiseMean, the
// implementation only considers 2D.
template <typename DeviceContext, typename T>
struct RowwiseMean2D {
  RowwiseMean2D(int left, int right, const platform::DeviceContext& dev_ctx);

  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* vec);
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class RowwiseMean2D<platform::CUDADeviceContext, T> {
 public:
  RowwiseMean2D(int left, int right, const platform::DeviceContext& dev_ctx)
      : left_(left), right_(right) {
    framework::DDim ones_dim({right_});
    divisor_.mutable_data<T>(ones_dim, dev_ctx.GetPlace());
    math::set_constant(dev_ctx, &divisor_, 1.0 / right);
  }
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* out) {
    math::GetBlas<platform::CUDADeviceContext, T>(context).GEMV(
        false, left_, right_, 1., input.data<T>(), divisor_.data<T>(), 0.,
        out->data<T>());
  }

 private:
  int left_;
  int right_;
  framework::Tensor divisor_;
};
#endif

template <typename T>
class RowwiseMean2D<platform::CPUDeviceContext, T> {
 public:
  RowwiseMean2D(int left, int right, const platform::DeviceContext& dev_ctx) {}

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* out) {
    row_mean_(context, input, out);
  }

 private:
  math::RowwiseMean<platform::CPUDeviceContext, T> row_mean_;
};

template <typename DeviceContext, typename T>
struct ColwiseSum2D {
  ColwiseSum2D(int left, int right, const platform::DeviceContext& dev_ctx);

  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* vec);
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class ColwiseSum2D<platform::CUDADeviceContext, T> {
 public:
  ColwiseSum2D(int left, int right, const platform::DeviceContext& dev_ctx)
      : left_(left), right_(right) {
    framework::DDim ones_dim({left_});
    divisor_.mutable_data<T>(ones_dim, dev_ctx.GetPlace());
    math::set_constant(dev_ctx, &divisor_, 1.0);
  }

  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* out) {
    math::GetBlas<platform::CUDADeviceContext, T>(context).GEMV(
        true, left_, right_, 1., input.data<T>(), divisor_.data<T>(), 0.,
        out->data<T>());
  }

 private:
  int left_;
  int right_;
  framework::Tensor divisor_;
};
#endif

template <typename T>
class ColwiseSum2D<platform::CPUDeviceContext, T> {
 public:
  ColwiseSum2D(int left, int right, const platform::DeviceContext& dev_ctx) {}

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* out) {
    col_wise_(context, input, out);
  }

 private:
  math::ColwiseSum<platform::CPUDeviceContext, T> col_wise_;
};

template <typename T>
struct SubAndSquareFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return (a - b) * (a - b); }
};

template <typename T>
struct DivAndSqrtFunctor {
  explicit DivAndSqrtFunctor(T epsilon) { epsilon_ = epsilon; }
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a / (sqrt(b + epsilon_));
  }

 private:
  T epsilon_;
};

template <typename T>
struct MulInvVarFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a * std::sqrt(1.0 / b);
  }
};

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class LayerNormDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream, const T* input,
                  std::vector<int> input_shape, const T* bias, const T* scale,
                  T* output, T* mean, T* variance, int begin_norm_axis,
                  float eps);
};
#endif

template <typename DeviceContext, typename T>
class LayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto x = *ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    const auto x_dims = x.dims();

    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    framework::DDim matrix_shape({left, right});

    x.Resize(matrix_shape);
    Tensor out;
    out.ShareDataWith(*y);
    out.Resize(matrix_shape);

#if defined(PADDLE_WITH_CUDA) || defined(_WIN32) || defined(__APPLE__) || \
    defined(__OSX__)
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    RowwiseMean2D<DeviceContext, T> row_mean(left, right, ctx.device_context());

    // get mean
    row_mean(dev_ctx, x, mean);

    // get variance
    ElementwiseComputeEx<SubAndSquareFunctor<T>, DeviceContext, T>(
        ctx, &x, mean, /*axis*/ 0, SubAndSquareFunctor<T>(), &out);
    row_mean(dev_ctx, out, var);

    // get x_norm
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
        ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), &out);
    ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
        ctx, &out, var, /*axis*/ 0,
        DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), &out);

    if (scale) {
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &out, scale, /*axis*/ 1, MulFunctor<T>(), &out);
    }
    if (bias) {
      ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
          ctx, &out, bias, /*axis*/ 1, AddFunctor<T>(), &out);
    }
#else
    PADDLE_ENFORCE_EQ(mean->numel(), left,
                      platform::errors::InvalidArgument(
                          "mean's length (%d) is not equal with expected (%d).",
                          mean->numel(), left));
    PADDLE_ENFORCE_EQ(var->numel(), left,
                      platform::errors::InvalidArgument(
                          "var's length (%d) is not equal with expected (%d).",
                          var->numel(), left));
    if (scale) {
      PADDLE_ENFORCE_EQ(
          scale->numel(), right,
          platform::errors::InvalidArgument(
              "scale's length (%d) is not equal with expected (%d).",
              scale->numel(), right));
    }
    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->numel(), right,
          platform::errors::InvalidArgument(
              "bias's length (%d) is not equal with expected (%d).",
              bias->numel(), right));
    }

    auto ker =
        jit::KernelFuncs<jit::LayerNormTuple<T>, platform::CPUPlace>::Cache()
            .At(right);
    ker(x.data<T>(), out.data<T>(), mean->data<T>(), var->data<T>(),
        scale ? scale->data<T>() : nullptr, bias ? bias->data<T>() : nullptr,
        static_cast<int>(left), static_cast<const float>(epsilon), right);
#endif
  }
};

template <typename DeviceContext, typename T>
class LayerNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto x = *ctx.Input<Tensor>("X");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto d_y = *ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x.dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    framework::DDim matrix_shape({left, right});

    d_y.Resize(matrix_shape);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    ColwiseSum2D<DeviceContext, T> colwise_sum(left, right,
                                               ctx.device_context());

    Tensor temp;
    Tensor temp_norm;
    if (d_scale || d_x) {
      x.Resize(matrix_shape);
      temp.mutable_data<T>(matrix_shape, ctx.GetPlace());

      temp_norm.mutable_data<T>(matrix_shape, ctx.GetPlace());
      // get x_norm
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), &temp_norm);
      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), &temp_norm);
    }

    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      colwise_sum(dev_ctx, d_y, d_bias);
    }
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &d_y, /*axis*/ 0, MulFunctor<T>(), &temp);
      colwise_sum(dev_ctx, temp, d_scale);
    }

    if (d_x) {
      framework::DDim vec_shape({left});
      d_x->mutable_data<T>(ctx.GetPlace());
      auto dx_dim = d_x->dims();
      Tensor temp_vec;
      temp_vec.mutable_data<T>(vec_shape, ctx.GetPlace());

      RowwiseMean2D<DeviceContext, T> row_mean(left, right,
                                               ctx.device_context());

      if (d_scale) {
        // dy_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, scale, /*axis*/ 1, MulFunctor<T>(), &temp);
        framework::TensorCopy(temp, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, temp, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &temp, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);
      } else {
        // dy_dx
        framework::TensorCopy(d_y, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, d_y, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);
      }
      // dy_var_dx
      row_mean(dev_ctx, temp, &temp_vec);
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &temp_vec, /*axis*/ 0, MulFunctor<T>(), &temp);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, d_x, &temp, /*axis*/ 0, SubFunctor<T>(), d_x);

      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, d_x, var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), d_x);
      d_x->Resize(dx_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle
