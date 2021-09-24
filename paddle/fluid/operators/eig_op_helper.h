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
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
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

#define DITO_SLICE_RANK_CASE(N)                      \
  case N: {                                          \
    EigenSliceWrapper<N>(&x, offset, extends, &ret); \
    break;                                           \
  }

template <typename DeviceContext, typename T>
void SolveLinearSystem(T* matrix_data, T* rhs_data, T* out_data, int order,
                       int rhs_cols, int batch) {
  using Treal = typename Eigen::NumTraits<T>::Real;

  std::complex<Treal>* matrix_data_ =
      reinterpret_cast<std::complex<Treal>*>(matrix_data);
  std::complex<Treal>* rhs_data_ =
      reinterpret_cast<std::complex<Treal>*>(rhs_data);
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

using InTensors = std::vector<const Tensor*>;
static std::vector<int> GetBroadcastShape(InTensors ins) {
  PADDLE_ENFORCE_EQ(ins.size(), 2, platform::errors::InvalidArgument(
                                       "GetBroadcastShape Receive 2 tensors"
                                       "but got [%d]",
                                       ins.size()));
  auto x_dim = ins[0]->dims();
  auto y_dim = ins[1]->dims();
  std::vector<int> broadcast_shape =
      (x_dim.size() > y_dim.size() ? framework::vectorize<int>(x_dim)
                                   : framework::vectorize<int>(y_dim));
  int rank_min = std::min(x_dim.size(), y_dim.size());
  int rank_x = x_dim.size();
  int rank_y = y_dim.size();
  int final_rank = broadcast_shape.size();
  for (int i = 1; i <= rank_min; ++i) {
    if (x_dim[rank_x - i] == y_dim[rank_y - i]) {
      broadcast_shape[final_rank - i] = x_dim[rank_x - i];
      continue;
    }
    if (x_dim[rank_x - i] == 1) {
      broadcast_shape[final_rank - i] = y_dim[rank_y - i];
      continue;
    }
    if (y_dim[rank_y - i] == 1) {
      broadcast_shape[final_rank - i] = x_dim[rank_x - i];
      continue;
    }
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Wrong Input Shape in broadcast operator: "
        "Input(X)'s shape must follow the broadcast rule with Input(Y)'s "
        "shape, but received [%s] (X) vs [%s] (Y).",
        x_dim, y_dim));
  }
  return broadcast_shape;
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
        x.dims(), context.GetPlace(),
        static_cast<size_t>(numel * sizeof(math::Real<T>)));
    auto* x_data = x.data<T>();
    auto for_range = GetForRange(numel);
    math::RealFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
    return out;
  }

  Tensor Div(const Tensor& x, const Tensor& y) {
    Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(framework::make_ddim(out_shape));
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
        context, &x, &y, -1, DivFunctor<T>(), &ret);
    return ret;
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

  // batch_diag
  Tensor Diag(const Tensor& x, int batch) {
    Tensor out;
    auto* x_data = x.data<math::Real<T>>();
    auto numel = x.numel();
    auto* out_data = out.mutable_data<math::Real<T>>(
        x.dims(), context.GetPlace(),
        static_cast<size_t>(numel * sizeof(math::Real<T>)));

    auto x_dims = x.dims();
    int num_dims = x_dims.size();
    std::vector<int> out_shape;

    for (int i = 0; i < num_dims - 1; ++i) {
      out_shape.push_back(x.dims()[i]);
    }
    out.Resize(framework::make_ddim(out_shape));
    int order = x.dims()[num_dims - 1];
    int stride_out = order * order;
    int stride_in = order + 1;
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < order; ++j) {
        out_data[i * order + j] = x_data[stride_out * i + stride_in * j];
      }
    }
    return out;
  }

  // a complex number x times a real number y, which is represented as (a+0j)
  Tensor RealMulComplex(const framework::Tensor& x,
                        const framework::Tensor& y) {
    framework::Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(framework::make_ddim(out_shape));
    ElementwiseComputeEx<RealMulComplexFunctor<T>, DeviceContext, T>(
        context, &x, &y, -1, RealMulComplexFunctor<T>(), &ret);
    return ret;
  }

  framework::Tensor Slice(const framework::Tensor& x, std::vector<int> axes,
                          std::vector<int> starts, std::vector<int> ends) {
    framework::Tensor ret;
    std::vector<int> new_axes = axes;
    std::vector<int> out_shape = framework::vectorize<int>(x.dims());
    size_t rank = out_shape.size();
    PADDLE_ENFORCE_EQ(
        axes.size(), starts.size(),
        platform::errors::InvalidArgument("Slice Operator Argument Invalided"));
    PADDLE_ENFORCE_EQ(
        ends.size(), starts.size(),
        platform::errors::InvalidArgument("Slice Operator Argument Invalided"));
    for (unsigned int i = 0; i < axes.size(); ++i) {
      int axis = axes[i];
      if (axis < 0) axis = rank + axis;
      new_axes[i] = axis;  // change negative to positive
      int st = starts[i];
      int ed = ends[i];
      PADDLE_ENFORCE_GT(ed, st,
                        platform::errors::InvalidArgument(
                            "C++ Slice Operation Not Support End < Start"));
      out_shape[axis] = ed - st;
    }
    std::vector<int> offset(rank), extends(rank);
    for (size_t i = 0; i < rank; ++i) {
      offset[i] = 0;
      extends[i] = x.dims()[i];
    }
    for (size_t i = 0; i < new_axes.size(); ++i) {
      offset[new_axes[i]] = starts[i];
      extends[new_axes[i]] = ends[i] - starts[i];
    }
    ret.Resize(framework::make_ddim(out_shape));
    ret.mutable_data<T>(context.GetPlace());
    switch (rank) {
      DITO_SLICE_RANK_CASE(1);
      DITO_SLICE_RANK_CASE(2);
      DITO_SLICE_RANK_CASE(3);
      DITO_SLICE_RANK_CASE(4);
      DITO_SLICE_RANK_CASE(5);
      DITO_SLICE_RANK_CASE(6);
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid Rank number, "
            "currently only support rank between 2~6"));
      }
    }
    return ret;
  }

 private:
  const framework::ExecutionContext& context;

  platform::ForRange<DeviceContext> GetForRange(int numel) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    return platform::ForRange<DeviceContext>(dev_ctx, numel);
  }

  template <size_t D>
  void EigenSliceWrapper(const framework::Tensor* in,
                         const std::vector<int>& start,
                         const std::vector<int>& end, framework::Tensor* out) {
    // Slice by call Eigen Tensor Function `.slice()`
    size_t rank = in->dims().size();
    PADDLE_ENFORCE_EQ(start.size(), rank,
                      platform::errors::InvalidArgument(
                          "EigenSliceWrapper function start "
                          "argument must have the same length as input rank."));
    PADDLE_ENFORCE_EQ(end.size(), rank,
                      platform::errors::InvalidArgument(
                          "EigenSliceWrapper function end "
                          "argument must have the same length as input rank."));
    auto eigen_place_ptr =
        context.template device_context<DeviceContext>().eigen_device();
    auto eigen_place = *eigen_place_ptr;
    auto out_t = framework::EigenTensor<T, D>::From(*out, out->dims());
    auto in_t = framework::EigenTensor<T, D>::From(*in, in->dims());
    Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
    for (size_t i = 0; i < D; i++) {
      offsets_32bit[i] = start[i];
      extents_32bit[i] = end[i];
    }
    EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place, framework::To32BitIndex(out_t),
        framework::To32BitIndex(in_t), offsets_32bit, extents_32bit);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
