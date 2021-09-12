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
//#include <Eigen/src/Core/util/Constants.h>
#include <assert.h>
#include <Eigen/Dense>
#include "Eigen/Core"
#include "Eigen/LU"
// #include <Eigen/Eigenvalues>
// #include <iostream>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
using InTensors = std::vector<const Tensor*>;
using OutTensors = std::vector<Tensor*>;
using Shape = std::vector<int>;
using OpName = std::string;

// T : paddle::complex
template <typename DeviceContext, typename T>
void SolveLinearSystem(T* matrix_data, T* rhs_data, T* out_data, int order,
                       int rhs_cols, int batch) {
  using Treal = typename Eigen::NumTraits<T>::Real;
  // std::complex<>

  // auto mat_dims = matrix->dims();
  // auto mat_num_dims = mat_dims.size();
  // int order = mat_dims[mat_num_dims - 1];
  // int rhs_cols = rhs->dims()[rhs->dims().size() - 1];

  // auto* matrix_data = matrix->data<T>();
  // auto* rhs_data = rhs->data<T>();
  // Tensor out;
  // auto* out_data = out.mutable_data<T>(rhs->dims(), ctx.GetPlace());

  std::complex<Treal>* matrix_data_ =
      reinterpret_cast<std::complex<Treal>*>(matrix_data);
  std::complex<Treal>* rhs_data_ =
      reinterpret_cast<std::complex<Treal>*>(rhs_data);
  // 输出会写入out_data吗！！！！
  std::complex<Treal>* out_data_ =
      reinterpret_cast<std::complex<Treal>*>(out_data);

  using Matrix = Eigen::Matrix<std::complex<Treal>, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<Matrix>;
  using OutputMatrixMap = Eigen::Map<Matrix>;

  for (int i = 0; i < batch; ++i) {
    auto input_matrix =
        InputMatrixMap(matrix_data_ + i * order * order, order, order);
    auto input_rhs =
        InputMatrixMap(rhs_data_ + i * order * rhs_cols, order, rhs_cols);
    auto output =
        OutputMatrixMap(out_data_ + i * order * rhs_cols, order, rhs_cols);

    Eigen::PartialPivLU<Matrix> lu_decomposition(order);
    lu_decomposition.compute(input_matrix);

    const Treal min_abs_piv =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    PADDLE_ENFORCE_GT(min_abs_piv, Treal(0),
                      platform::errors::InvalidArgument(
                          "Something's wrong with SolveLinearSystem. "));

    output = lu_decomposition.solve(input_rhs);
  }
}

template <typename Tout, typename Tbase>
void SolveComplexInput(Tout* x_data, Tout* eigenvalues_data,
                       Tout* eigenvectors_data, int batch, int order) {
  // LOG(INFO) << "⭕️ Complex Input 0";
  //// 输入没错
  // LOG(INFO) << "x_data[0]: " << x_data[0] << ", "<<x_data[1]<< ",
  // "<<x_data[2]<< ", "<<x_data[3]<< ", "<<x_data[4]<< ", "<<x_data[5];
  // cast paddle::complex<> to std::complex<>
  std::complex<Tbase>* x_data_ = reinterpret_cast<std::complex<Tbase>*>(x_data);
  std::complex<Tbase>* eigenvalues_data_ =
      reinterpret_cast<std::complex<Tbase>*>(eigenvalues_data);
  std::complex<Tbase>* eigenvectors_data_ =
      reinterpret_cast<std::complex<Tbase>*>(eigenvectors_data);

  using InputMatrix = Eigen::Matrix<std::complex<Tbase>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<const InputMatrix>;
  using OutputMatrix = Eigen::Matrix<std::complex<Tbase>, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
  using OutputMatrixMap = Eigen::Map<OutputMatrix>;

  for (int i = 0; i < batch; ++i) {
    auto input = InputMatrixMap(x_data_ + i * order * order, order, order);
    auto output_values =
        OutputMatrixMap(eigenvalues_data_ + i * order, 1, order);
    auto output_vectors =
        OutputMatrixMap(eigenvectors_data_ + i * order * order, order, order);

    // LOG(INFO) << "循环中的输入[]: " << (x_data + i * order * order)[0]<<",
    // "<< (x_data + i * order * order)[1]<<", "<< (x_data + i * order *
    // order)[2]<<", "<< (x_data + i * order * order)[3]<<", "<< (x_data + i *
    // order * order)[4];
    Eigen::ComplexEigenSolver<OutputMatrix> eig_solver(
        input, Eigen::ComputeEigenvectors);
    PADDLE_ENFORCE_EQ(eig_solver.info(), Eigen::Success,
                      platform::errors::InvalidArgument(
                          "Eigen decomposition was not "
                          "successful. The input might not be valid."));
    output_values = eig_solver.eigenvalues().transpose();
    output_vectors = eig_solver.eigenvectors();
  }
  // LOG(INFO) << "⭕️ Complex Input 1";
}

// Tin=float          Tout=complexfloat   Tbase=float
// Tin=complexfloat   Tout=complexfloat   Tbase=float
template <typename Tout, typename Tbase>
void SolveNonComplexInput(Tbase* x_data, Tout* eigenvalues_data,
                          Tout* eigenvectors_data, int batch, int order) {
  // LOG(INFO) << "🛑 NonComplex Input 0";
  //// 输入没错
  // LOG(INFO) << "x_data[0]: " << x_data[0] << ", "<<x_data[1]<< ",
  // "<<x_data[2]<< ", "<<x_data[3]<< ", "<<x_data[4]<< ", "<<x_data[5];
  // cast paddle::complex to std::complex
  std::complex<Tbase>* eigenvalues_data_ =
      reinterpret_cast<std::complex<Tbase>*>(eigenvalues_data);
  std::complex<Tbase>* eigenvectors_data_ =
      reinterpret_cast<std::complex<Tbase>*>(eigenvectors_data);

  using InputMatrix =
      Eigen::Matrix<Tbase, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<const InputMatrix>;
  using OutputMatrix = Eigen::Matrix<std::complex<Tbase>, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
  using OutputMatrixMap = Eigen::Map<OutputMatrix>;

  for (int i = 0; i < batch; ++i) {
    auto input = InputMatrixMap(x_data + i * order * order, order, order);
    auto output_values =
        OutputMatrixMap(eigenvalues_data_ + i * order, 1, order);
    auto output_vectors =
        OutputMatrixMap(eigenvectors_data_ + i * order * order, order, order);

    /// 输入没错
    // LOG(INFO) << "循环中的输入[]: " << (x_data + i * order * order)[0]<<",
    // "<< (x_data + i * order * order)[1]<<", "<< (x_data + i * order *
    // order)[2]<<", "<< (x_data + i * order * order)[3]<<", "<< (x_data + i *
    // order * order)[4];
    Eigen::ComplexEigenSolver<OutputMatrix> eig_solver(
        input, Eigen::ComputeEigenvectors);
    PADDLE_ENFORCE_EQ(eig_solver.info(), Eigen::Success,
                      platform::errors::InvalidArgument(
                          "Eigen decomposition was not "
                          "successful. The input might not be valid."));
    output_values = eig_solver.eigenvalues().transpose();
    output_vectors = eig_solver.eigenvectors();
  }
  // LOG(INFO) << "🛑 NonComplex Input 1";
}

using Tensor = framework::Tensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

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

  Tensor Real(const Tensor& x) {
    Tensor out;
    auto numel = x.numel();
    auto* out_data = out.mutable_data<math::Real<T>>(
        context.GetPlace(), static_cast<size_t>(numel * sizeof(math::Real<T>)));
    auto* x_data = x.data<T>();
    auto for_range = GetForRange(numel);
    math::RealFunctor<T> functor(x_data, out_data, numel);
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
    LOG(INFO) << "⭕️ SubBroadcast out_dims: " << out.dims();
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
    LOG(INFO) << "⭕️ Unsqueeze out_dims: " << out.dims();
    return out;
  }

  // batch_diag
  Tensor Diag(const Tensor& x, int batch) {
    Tensor out;
    auto* out_data = out.mutable_data<T>(x.dims(), context.GetPlace());
    auto* x_data = x.data<T>();
    auto x_dims = x.dims();
    LOG(INFO) << "🌸 x_dims: " << x_dims;
    int num_dims = x_dims.size();  //
    LOG(INFO) << "🌸 num_dims: " << num_dims;
    std::vector<int> out_shape;

    for (int i = 0; i < num_dims - 1; ++i) {
      out_shape.push_back(x.dims()[i]);
    }
    out.Resize(framework::make_ddim(out_shape));
    auto out_dims = out.dims();

    int order = x.dims()[num_dims - 1];
    // int batch = BatchCount(x);
    auto out_length = batch * order;
    LOG(INFO) << "out_dims[0]: " << out_length;

    int stride_out = order * order;
    int stride_in = order + 1;
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < order; ++j) {
        out_data[i * order + j] = x_data[stride_out * i + stride_in * j];
      }
    }
    LOG(INFO) << "🌸 out_dims: " << out_dims;
    return out;
  }

 private:
  const framework::ExecutionContext& context;

  platform::ForRange<DeviceContext> GetForRange(int numel) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    return platform::ForRange<DeviceContext>(dev_ctx, numel);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
