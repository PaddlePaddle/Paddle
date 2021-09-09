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

#pragma once
#include <Eigen/Dense>
#include "Eigen/Core"
#include "paddle/fluid/framework/op_registry.h"
// #include "paddle/fluid/operators/eigh_helper.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using InputMatrixMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using OutputMatrixMap = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename ValueType>
inline void BatchEigenvalues(ValueType* x_data, ValueType* eigenvalues_data,
                             ValueType* eigenvectors_data, int batches,
                             int rows, int cols) {
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = InputMatrixMap<ValueType>(x_data + i * stride, rows, cols);
    auto eigenvalues =
        OutputMatrixMap<ValueType>(eigenvalues_data + i * rows, 1, rows);
    auto eigenvectors =
        OutputMatrixMap<ValueType>(eigenvectors_data + i * stride, rows, cols);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<
        ValueType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    PADDLE_ENFORCE_EQ(
        eigen_solver.info(), Eigen::Success,
        platform::errors::InvalidArgument(
            "Self Adjoint Eigen decomposition is not successful. "
            "The %d-th input matrice might not be not be positive definite.",
            i));
    eigenvalues = eigen_solver.eigenvalues().transpose();
    eigenvectors = eigen_solver.eigenvectors().transpose();
  }
}

template <typename T, typename ValueType>
inline void BatchComplexValues(T* x_data, ValueType* eigenvalues_data,
                               T* eigenvectors_data, int batches, int rows,
                               int cols) {
  using Complex = std::complex<ValueType>;
  Complex* input = reinterpret_cast<Complex*>(x_data);
  Complex* eigenvectors_data_ = reinterpret_cast<Complex*>(eigenvectors_data);

  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = InputMatrixMap<Complex>(input + i * stride, rows, cols);
    auto eigenvalues =
        OutputMatrixMap<ValueType>(eigenvalues_data + i * rows, 1, rows);
    auto eigenvectors =
        OutputMatrixMap<Complex>(eigenvectors_data_ + i * stride, rows, cols);

    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    PADDLE_ENFORCE_EQ(
        eigen_solver.info(), Eigen::Success,
        platform::errors::InvalidArgument(
            "Self Adjoint Eigen decomposition is not successful. "
            "The %d-th input matrice might not be not be positive definite.",
            i));

    eigenvalues = eigen_solver.eigenvalues().transpose();
    eigenvectors = eigen_solver.eigenvectors().transpose();
  }
}

template <typename T, typename ValueType>
struct DiagAndCopyFunctor {
  DiagAndCopyFunctor(const int m, const int n, const int num_lower_diags,
                     const int num_upper_diags, const ValueType* scale,
                     const T* input, T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        scale_(scale),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = input_[index];
    } else if (col == band_end - 1) {
      output_[index] = static_cast<T>(scale_[index % m_]);
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const ValueType* scale_;
  const T* input_;
  T* output_;
};

template <typename DeviceContext, typename T, typename ValueType>
struct DeviceIndependenceTensorOperations {
  explicit DeviceIndependenceTensorOperations(
      const framework::ExecutionContext& context)
      : context(context) {}

  Tensor DiagFill(const int m, const int n, const int num_lower_diags,
                  const int num_upper_diags, const Tensor& scale,
                  const Tensor& input) {
    Tensor out;
    auto for_range = GetForRange(input.numel());
    DiagAndCopyFunctor<T, ValueType> diag_and_copy_functor(
        m, n, num_lower_diags, num_upper_diags, scale.data<ValueType>(),
        input.data<T>(), out.mutable_data<T>(input.dims(), input.place()));
    for_range(diag_and_copy_functor);
    return out;
  }

  Tensor Matmul(const Tensor& mat_a, const Tensor& mat_b) {
    Tensor out;
    out.mutable_data<T>(mat_a.dims(), context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto no_trans_desc = math::CreateMatrixDescriptor(mat_a.dims(), 0, false);
    blas.MatMul(mat_a, no_trans_desc, mat_b, no_trans_desc, T(1), &out, T(0));
    return out;
  }

  // transpose the last two dimision
  Tensor Transpose(const Tensor& x) {
    Tensor out;
    auto& dims = x.dims();
    out.mutable_data<T>(dims, context.GetPlace());
    std::vector<int> axis(dims.size() - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dims.size() - 1, dims.size() - 2});
    auto& dev_ctx = context.template device_context<DeviceContext>();
    TransCompute<DeviceContext, T>(dims.size(), dev_ctx, x, &out, axis);
    return out;
  }

  Tensor Conj(const Tensor& x) {
    Tensor out;
    auto* out_data = out.mutable_data<T>(x.dims(), context.GetPlace());
    auto* x_data = x.data<T>();
    auto for_range = GetForRange(x.numel());
    math::ConjFunctor<T> functor(x_data, x.numel(), out_data);
    for_range(functor);
    return out;
  }

  Tensor Mul(const Tensor& x, float a) {
    Tensor out;
    out.mutable_data<T>(x.dims(), context.GetPlace());
    auto x_vector = EigenVector<T>::Flatten(x);
    auto out_vector = EigenVector<T>::Flatten(out);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    out_vector.device(place) = x_vector * static_cast<T>(a);
    return out;
  }

  Tensor Div(const Tensor& x, const Tensor& y) {
    Tensor out;
    out.mutable_data<T>(x.dims(), context.GetPlace());
    auto x_vector = EigenVector<T>::Flatten(x);
    auto y_vector = EigenVector<ValueType>::Flatten(y);
    auto out_vector = EigenVector<T>::Flatten(out);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    out_vector.device(place) = x_vector / y_vector;
    return out;
  }

  Tensor Sub(const Tensor& x, const Tensor& y) {
    Tensor out;
    out.mutable_data<T>(x.dims(), context.GetPlace());
    auto x_vector = EigenVector<T>::Flatten(x);
    auto y_vector = EigenVector<T>::Flatten(y);
    auto out_vector = EigenVector<T>::Flatten(out);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    out_vector.device(place) = x_vector - y_vector;
    return out;
  }

  Tensor SubBroadcast(const Tensor& x, const Tensor& y, int batch_size, int m) {
    Tensor out;
    auto& dims = x.dims();
    std::vector<int> vec_dim;
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    if (batch_size > 1) {
      vec_dim.push_back(batch_size);
      vec_dim.push_back(dims[dims.size() - 1]);
      vec_dim.push_back(dims[dims.size() - 1]);
      out.mutable_data<ValueType>(framework::make_ddim(vec_dim),
                                  context.GetPlace());
      auto x_tensor = EigenTensor<ValueType, 3>::From(x);
      auto y_tensor = EigenTensor<ValueType, 3>::From(y);
      auto out_tensor = EigenTensor<ValueType, 3>::From(out);
      Eigen::DSizes<int, 3> a_bcast_dims(1, m, 1);
      Eigen::DSizes<int, 3> b_bcast_dims(1, 1, m);
      out_tensor.device(place) =
          x_tensor.broadcast(a_bcast_dims) - y_tensor.broadcast(b_bcast_dims);
    } else {
      vec_dim.push_back(dims[dims.size() - 1]);
      vec_dim.push_back(dims[dims.size() - 1]);
      out.mutable_data<ValueType>(framework::make_ddim(vec_dim),
                                  context.GetPlace());
      auto x_tensor = EigenTensor<ValueType, 2>::From(x);
      auto y_tensor = EigenTensor<ValueType, 2>::From(y);
      auto out_tensor = EigenTensor<ValueType, 2>::From(out);
      Eigen::DSizes<int, 2> a_bcast_dims(m, 1);
      Eigen::DSizes<int, 2> b_bcast_dims(1, m);
      out_tensor.device(place) =
          x_tensor.broadcast(a_bcast_dims) - y_tensor.broadcast(b_bcast_dims);
    }
    return out;
  }

  const Tensor Unsqueeze(const framework::Tensor& x, int axis = 0) {
    framework::Tensor out;
    out.ShareDataWith(x);
    std::vector<int> out_shape = framework::vectorize<int>(x.dims());
    if (axis >= 0) {
      auto index = (out_shape.begin() + axis);
      out_shape.insert(index, 1);
    } else if (axis < 0) {
      auto index = (out_shape.end() + axis + 1);
      out_shape.insert(index, 1);
    }
    out.Resize(framework::make_ddim(out_shape));
    return out;
  }

 private:
  const framework::ExecutionContext& context;
  platform::ForRange<DeviceContext> GetForRange(int numel) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    return platform::ForRange<DeviceContext>(dev_ctx, numel);
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& input_var = *ctx.Input<Tensor>("X");
    auto& output_w_var = *ctx.Output<Tensor>("Eigenvalues");
    auto& output_v_var = *ctx.Output<Tensor>("Eigenvectors");

    std::string lower = ctx.Attr<std::string>("UPLO");
    auto dims = input_var.dims();
    auto output_value_dim = output_w_var.dims();

    int64_t batch_size = 1;
    int dim_size = dims.size();
    for (int64_t i = 0; i < dim_size - 2; i++) {
      batch_size *= dims[i];
    }
    auto dito =
        DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(ctx);
    Tensor input;
    TensorCopy(input_var, ctx.GetPlace(), &input);
    if (lower == "U") {
      input = dito.Transpose(input_var);
    }
    int rows = dims[dims.size() - 2];
    int cols = dims[dims.size() - 1];

    auto* value_data =
        output_w_var.mutable_data<ValueType>(output_value_dim, ctx.GetPlace());

    if (framework::IsComplexType(input.type())) {
      auto* x_data = input.data<T>();
      auto* vector_data = output_v_var.mutable_data<T>(dims, ctx.GetPlace());
      BatchComplexValues<T, ValueType>(x_data, value_data, vector_data,
                                       batch_size, rows, cols);
    } else {
      auto* x_data = input.data<ValueType>();
      auto* vector_data =
          output_v_var.mutable_data<ValueType>(dims, ctx.GetPlace());
      BatchEigenvalues<ValueType>(x_data, value_data, vector_data, batch_size,
                                  rows, cols);
    }
    output_v_var = dito.Transpose(output_v_var);
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad.mutable_data<T>(ctx.GetPlace());
    auto& output_w_var = *ctx.Input<Tensor>("Eigenvalues");
    auto& output_v_var = *ctx.Input<Tensor>("Eigenvectors");
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvalues"));
    auto& output_v_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvectors"));

    auto& dims = output_v_var.dims();
    int batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    int cols = dims[dims.size() - 1];
    auto dito =
        DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(ctx);

    auto tV = dito.Transpose(dito.Conj(output_v_var));
    auto w_sub =
        dito.SubBroadcast(dito.Unsqueeze(output_w_var, -2),
                          dito.Unsqueeze(output_w_var, -1), batch_size, cols);

    Tensor result = dito.Matmul(tV, output_v_grad);
    auto res_trans = dito.Transpose(result);
    res_trans = dito.Conj(res_trans);
    result = dito.Sub(result, res_trans);
    result = dito.Mul(result, 0.5);
    result = dito.Div(result, w_sub);
    result = dito.DiagFill(cols, cols, cols, 0, output_w_grad, result);
    x_grad = dito.Matmul(output_v_var, dito.Matmul(result, tV));
  }
};

}  // namespace operators
}  // namespace paddle
