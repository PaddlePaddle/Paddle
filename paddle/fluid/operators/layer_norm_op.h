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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

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

#ifdef PADDLE_WITH_CUDA
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
        out.data<T>());
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

#ifdef PADDLE_WITH_CUDA
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
        out.data<T>());
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
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
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

#ifdef __AVX__
#ifndef AVX_FLOAT_BLOCK
#define AVX_FLOAT_BLOCK 8
#endif
    if (std::is_same<T, float>::value && (right >= AVX_FLOAT_BLOCK)) {
      T* in_buf;
      T* out_buf;
      const T* in_buf_scale = scale->data<T>();
      const T* in_buf_bias = bias->data<T>();
      int height = left;
      int size = right;
      __m256 size_vec = _mm256_set1_ps(size);
      __m256 epsilon_vec = _mm256_set1_ps(epsilon);
      const int block = AVX_FLOAT_BLOCK;
      const int rest = size % block;
      const int rest_mask =
          ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
      __m256i mask_vec = _mm256_set_epi32(
          rest_mask & 0x80 ? 0xffffffff : 0, rest_mask & 0x40 ? 0xffffffff : 0,
          rest_mask & 0x20 ? 0xffffffff : 0, rest_mask & 0x10 ? 0xffffffff : 0,
          rest_mask & 0x8 ? 0xffffffff : 0, rest_mask & 0x4 ? 0xffffffff : 0,
          rest_mask & 0x2 ? 0xffffffff : 0, rest_mask & 0x1 ? 0xffffffff : 0);
      int end = size - rest;

      PADDLE_ENFORCE_EQ(mean->numel(), left);
      PADDLE_ENFORCE_EQ(var->numel(), left);
      PADDLE_ENFORCE_EQ(scale->numel(), right);
      PADDLE_ENFORCE_EQ(bias->numel(), right);

      __m256 sum;
      __m256 mean_vec, var_vec;
      __m128 hi, lo;
      __m256 last_vec;
      __m256 tmp;
      size_t offset;
      size_t j;

      for (int i = 0; i < height; ++i) {
        in_buf = x.data<T>();
        out_buf = mean->data<T>();
        offset = i * size;

        // get mean
        sum = _mm256_setzero_ps();
        for (j = offset; j < end + offset; j += block) {
          sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)in_buf + j));
        }
        if (rest != 0) {
          j = offset + size - block;
          tmp = _mm256_loadu_ps((const float*)in_buf + j);
          tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp, (__m256)mask_vec);
          sum = _mm256_add_ps(sum, tmp);
        }
        hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm256_extractf128_ps(sum, 0);
        sum = _mm256_add_ps(
            sum, _mm256_insertf128_ps(
                     _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        mean_vec = _mm256_div_ps(sum, size_vec);
        out_buf[i] = *reinterpret_cast<float*>(&mean_vec);

        // get variance
        out_buf = var->data<T>();
        sum = _mm256_setzero_ps();
        for (j = offset; j < end + offset; j += block) {
          tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)in_buf + j),
                              mean_vec);
          tmp = _mm256_mul_ps(tmp, tmp);
          sum = _mm256_add_ps(sum, tmp);
        }
        if (rest != 0) {
          j = offset + size - block;
          tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)in_buf + j),
                              mean_vec);
          tmp = _mm256_mul_ps(tmp, tmp);
          tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp, (__m256)mask_vec);
          sum = _mm256_add_ps(sum, tmp);
        }
        hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm256_extractf128_ps(sum, 0);
        sum = _mm256_add_ps(
            sum, _mm256_insertf128_ps(
                     _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        var_vec = _mm256_div_ps(sum, size_vec);
        out_buf[i] = *reinterpret_cast<float*>(&var_vec);

        // get x_norm
        out_buf = out.data<T>();
        for (j = offset; j < end + offset; j += block) {
          tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)in_buf + j),
                              mean_vec);
          tmp = _mm256_div_ps(
              tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
          _mm256_storeu_ps(reinterpret_cast<float*>(out_buf) + j, tmp);
        }
        if (rest != 0) {
          j = offset + size - block;
          tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)in_buf + j),
                              mean_vec);
          tmp = _mm256_div_ps(
              tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
          _mm256_storeu_ps(reinterpret_cast<float*>(out_buf) + j, tmp);
        }

        in_buf = out.data<T>();
        if (scale) {
          if (rest != 0) {
            j = offset + size - block;
            last_vec = _mm256_loadu_ps((const float*)in_buf + j);
          }
          for (j = offset; j < end + offset; j += block) {
            _mm256_storeu_ps(
                reinterpret_cast<float*>(out_buf) + j,
                _mm256_mul_ps(
                    _mm256_loadu_ps((const float*)in_buf + j),
                    _mm256_loadu_ps((const float*)in_buf_scale + j - offset)));
          }
          if (rest != 0) {
            j = offset + size - block;
            _mm256_storeu_ps(
                reinterpret_cast<float*>(out_buf) + j,
                _mm256_mul_ps(
                    last_vec,
                    _mm256_loadu_ps((const float*)in_buf_scale + j - offset)));
          }
        }

        if (bias) {
          if (rest != 0) {
            j = offset + size - block;
            last_vec = _mm256_loadu_ps((const float*)in_buf + j);
          }
          for (j = offset; j < end + offset; j += block) {
            _mm256_storeu_ps(
                reinterpret_cast<float*>(out_buf) + j,
                _mm256_add_ps(
                    _mm256_loadu_ps((const float*)in_buf + j),
                    _mm256_loadu_ps((const float*)in_buf_bias + j - offset)));
          }
          if (rest != 0) {
            j = offset + size - block;
            _mm256_storeu_ps(
                reinterpret_cast<float*>(out_buf) + j,
                _mm256_add_ps(
                    last_vec,
                    _mm256_loadu_ps((const float*)in_buf_bias + j - offset)));
          }
        }
      }

    } else {
#endif
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      RowwiseMean2D<DeviceContext, T> row_mean(left, right,
                                               ctx.device_context());

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
#ifdef __AVX__
    }
#endif
  }
};

template <typename DeviceContext, typename T>
class LayerNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto x = *ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
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

      if (!(bias && scale)) {
        temp_norm.ShareDataWith(*y);
        temp_norm.Resize(matrix_shape);
      } else {
        temp_norm.mutable_data<T>(matrix_shape, ctx.GetPlace());
        // get x_norm
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), &temp_norm);
        ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
            ctx, &temp_norm, var, /*axis*/ 0,
            DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), &temp_norm);
      }
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
