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

#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/diag_op.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
namespace math {
using Tensor = framework::Tensor;
using InTensors = std::vector<const Tensor*>;
using OutTensors = std::vector<Tensor*>;
using OpName = std::string;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
void EigenSvd(const T* X, T* U, T* VH, T* S, int rows, int cols,
              int full = false) {
  auto flag = Eigen::DecompositionOptions::ComputeThinU |
              Eigen::DecompositionOptions::ComputeThinV;
  if (full) {
    flag = Eigen::DecompositionOptions::ComputeFullU |
           Eigen::DecompositionOptions::ComputeFullV;
  }
  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd(2, 2, flag);
  /*NOTE(xiongkun03) Eigen::Matrix API need non-const pointer.*/
  T* input = const_cast<T*>(X);
  auto m = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      input, rows, cols);
  svd.compute(m);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_trans =
      svd.matrixV().transpose();
  memcpy(U, svd.matrixU().data(), svd.matrixU().size() * sizeof(T));
  memcpy(VH, V_trans.data(), V_trans.size() * sizeof(T));
  memcpy(S, svd.singularValues().data(),
         svd.singularValues().size() * sizeof(T));
}

template <typename T>
void BatchSvd(const T* X, T* U, T* VH, T* S, int rows, int cols, int batches,
              int full = false) {
  int stride = rows * cols;
  int k = std::min(rows, cols);
  int stride_u = full ? rows * rows : k * rows;
  int stride_v = full ? cols * cols : k * cols;
  for (int i = 0; i < batches; ++i) {
    EigenSvd<T>(X + i * stride, U + i * stride_u, VH + i * stride_v, S + i * k,
                rows, cols, full);
  }
  return;
}

template <typename T>
struct PowFunctor {
  PowFunctor(const T* input, T* output, int64_t numel, T exp)
      : input_(input), output_(output), numel_(numel), exp_(exp) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = pow(input_[idx], exp_);
  }
  const T* input_;
  T* output_;
  int64_t numel_;
  T exp_;
};

template <typename T>
struct RealMulComplexFunctor {
  // x: complex number (a+bj)
  // y: complex number (c+0j) pretend to be a real number
  // out: complex number (ac+bcj)
  inline HOSTDEVICE T operator()(T x, T y) {
    PADDLE_ENFORCE_LT(y.imag, 1e-6, platform::errors::InvalidArgument(
                                        "The image part of y must to be 0"
                                        "but got [%d]",
                                        y.imag));
    return platform::complex<phi::dtype::Real<T>>(x.real * y.real,
                                                  x.imag * y.real);
  }
};

static std::vector<int> GetBroadcastShape(InTensors ins) {
  PADDLE_ENFORCE_EQ(ins.size(), 2, platform::errors::InvalidArgument(
                                       "GetBroadcastShape Receive 2 tensors"
                                       "but got [%d]",
                                       ins.size()));
  auto x_dim = ins[0]->dims();
  auto y_dim = ins[1]->dims();
  std::vector<int> broadcast_shape =
      (x_dim.size() > y_dim.size() ? phi::vectorize<int>(x_dim)
                                   : phi::vectorize<int>(y_dim));
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

static inline framework::DDim ComputeAndCheckShapeForConcatOp(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims,
    const size_t axis) {
  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(), out_dims.size(),
                      platform::errors::InvalidArgument(
                          "The shape of input[0] and input[%d] "
                          "is expected to be equal."
                          "But received input[0]'s shape = "
                          "[%s], input[%d]'s shape = [%s].",
                          i, inputs_dims[0], i, inputs_dims[i]));
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        if (is_runtime) {
          out_dims[axis] += inputs_dims[i][j];
        } else {
          if (inputs_dims[i][j] == -1 || out_dims[j] == -1) {
            out_dims[axis] = -1;
          } else {
            out_dims[axis] += inputs_dims[i][j];
          }
        }
      } else {
        bool check_shape =
            is_runtime || (inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0);
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j], inputs_dims[i][j],
                            platform::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j, i, inputs_dims[0], i, inputs_dims[i]));
        }
        if (!is_runtime && out_dims[j] == -1 && inputs_dims[i][j] > 0) {
          out_dims[j] = inputs_dims[i][j];
        }
      }
    }
  }
  return out_dims;
}

static inline int64_t ComputeAxisForConcatOp(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank, true,
      platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d", -rank,
          rank, axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

// Prepared for the broadcast operation
static std::vector<int64_t> get_broadcast_batch_portion(
    std::vector<int64_t> x, std::vector<int64_t> y) {
  size_t size_x = x.size();
  size_t size_y = y.size();
  size_t size = std::max(size_x, size_y);
  std::vector<int64_t> batchPortion(size);

  ptrdiff_t i = (ptrdiff_t)size - 1;
  for (; i >= 0; --i) {
    ptrdiff_t offset = size - i - 1;
    ptrdiff_t dim_x = size_x - offset - 1;
    ptrdiff_t dim_y = size_y - offset - 1;
    int64_t x_size = (dim_x >= 0) ? x[dim_x] : 1;
    int64_t y_size = (dim_y >= 0) ? y[dim_y] : 1;

    PADDLE_ENFORCE_EQ(
        (x_size == y_size || x_size == 1 || y_size == 1), true,
        platform::errors::PreconditionNotMet(
            "The size of tensor x (%d) must match the size of tensor y "
            "(%d) at non-singleton dimension %d.",
            x_size, y_size, i));

    batchPortion[i] = x_size != 1 ? x_size : y_size;
  }
  return batchPortion;
}

#define DITO_TRANSPOSE_RANK_CASE(N)                   \
  case N: {                                           \
    phi::funcs::Transpose<DeviceContext, T, N> trans; \
    trans(dev_ctx, x, &ret, axis);                    \
    break;                                            \
  }

#define DITO_SLICE_RANK_CASE(N)                      \
  case N: {                                          \
    EigenSliceWrapper<N>(&x, offset, extends, &ret); \
    break;                                           \
  }

template <typename T, typename ValueType>
struct DiagAndFillFunctor {
  DiagAndFillFunctor(const int m, const int n, const int num_lower_diags,
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

 private:
  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const ValueType* scale_;
  const T* input_;
  T* output_;
};

template <typename DeviceContext, typename T, typename ValueType = T>
struct DeviceIndependenceTensorOperations {
  // 1. Device indenpendence, for kernel reuse.
  // 2. Input and output is always tensor type.
  // 3. output Tensor is alway allocated
  // 4. Basic Tensor operator is supported
  // 5. The Reused Operator Kernel should only be considered as
  //    a wrap function
  using NameInTensorMap =
      std::map<std::string, std::vector<const framework::Tensor*>>;
  using NameOutTensor = std::vector<std::string>;

  explicit DeviceIndependenceTensorOperations(
      const framework::ExecutionContext& context)
      : context(context) {}

  framework::Tensor Pow(const framework::Tensor& x, T exp) {
    framework::Tensor out;
    auto for_range = GetForRange(x.numel());
    int numel = x.numel();
    PowFunctor<T> functor(x.data<T>(), out.mutable_data<T>(x.dims(), x.place()),
                          numel, exp);
    for_range(functor);
    return out;
  }
  framework::Tensor Matmul(const framework::Tensor& mat_a,
                           const framework::Tensor& mat_b, bool trans_a = false,
                           bool trans_b = false) {
    framework::Tensor ret;
    auto a_dim = mat_a.dims();
    auto b_dim = mat_b.dims();
    std::vector<int> x_vec = phi::vectorize<int>(a_dim);
    x_vec[x_vec.size() - 2] = a_dim[a_dim.size() - (trans_a ? 1 : 2)];
    x_vec[x_vec.size() - 1] = b_dim[b_dim.size() - (trans_b ? 2 : 1)];
    ret.Resize(phi::make_ddim(x_vec));
    ret.mutable_data<T>(context.GetPlace());
    auto blas = GetBlas();
    auto mat_a_discrib = phi::funcs::CreateMatrixDescriptor(a_dim, 0, trans_a);
    auto mat_b_discrib = phi::funcs::CreateMatrixDescriptor(b_dim, 0, trans_b);
    blas.MatMul(mat_a, mat_a_discrib, mat_b, mat_b_discrib, T(1.0), &ret,
                T(0.0));
    return ret;
  }

  framework::Tensor Transpose(const framework::Tensor& x) {
    // transpose the last two dimision
    framework::Tensor ret;
    auto x_dim = x.dims();
    auto x_vec = phi::vectorize<int>(x_dim);
    int rank = x_vec.size();
    std::swap(x_vec[rank - 1], x_vec[rank - 2]);
    std::vector<int> out_shape = x_vec;
    std::vector<int> axis(rank);
    for (int i = 0; i < rank; ++i) {
      axis[i] = i;
    }
    std::swap(axis[rank - 1], axis[rank - 2]);
    auto& dev_ctx = context.template device_context<DeviceContext>();
    ret.Resize(phi::make_ddim(x_vec));
    ret.mutable_data<T>(context.GetPlace());
    switch (rank) {
      DITO_TRANSPOSE_RANK_CASE(2);
      DITO_TRANSPOSE_RANK_CASE(3);
      DITO_TRANSPOSE_RANK_CASE(4);
      DITO_TRANSPOSE_RANK_CASE(5);
      DITO_TRANSPOSE_RANK_CASE(6);
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid Rank number, "
            "currently only support rank between 2~6"));
      }
    }
    return ret;
  }
  framework::Tensor Diag(const framework::Tensor& x, int offset = 0,
                         // FIXME  link error
                         int padding_value = 0) {
    PADDLE_ENFORCE_EQ(padding_value, 0,
                      platform::errors::InvalidArgument(
                          "Current diag only support padding_value = 0"));
    PADDLE_ENFORCE_EQ(offset, 0,
                      platform::errors::InvalidArgument(
                          "Current diag only support offset = 0,"
                          "you can use DiagOp instead(not recommend)"));

    framework::Tensor ret;
    int x_rank = x.dims().size();
    std::vector<int> out_shape;
    if (x_rank == 2) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Current diag only support vector"
          "-> diagonalized matrix, not support matrix -> vector,"
          " Use DiagOp instead."));
    } else if (x_rank == 1) {
      out_shape.push_back(x.dims()[0]);
      out_shape.push_back(x.dims()[0]);
    } else {
      PADDLE_THROW(
          platform::errors::InvalidArgument("Rank must less or equal than 2"));
    }
    ret = Fill({out_shape[0], out_shape[0]}, 0.0);
    T* output = ret.mutable_data<T>(context.GetPlace());
    auto for_range = GetForRange(x.numel());
    for_range(DiagFunctor<T>(x.data<T>(), x.numel(), output));
    return ret;
  }

  // batch_diag for CPU only
  Tensor BatchDiag(const Tensor& x, int batch) {
    Tensor out;
    auto* x_data = x.data<phi::dtype::Real<T>>();
    auto numel = x.numel();
    auto* out_data = out.mutable_data<phi::dtype::Real<T>>(
        x.dims(), context.GetPlace(),
        static_cast<size_t>(numel * sizeof(phi::dtype::Real<T>)));

    auto x_dims = x.dims();
    int num_dims = x_dims.size();
    std::vector<int> out_shape;

    for (int i = 0; i < num_dims - 1; ++i) {
      out_shape.push_back(x.dims()[i]);
    }
    out.Resize(phi::make_ddim(out_shape));
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
  Tensor RealMulComplex(const Tensor& x, const Tensor& y) {
    framework::Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(phi::make_ddim(out_shape));
    ElementwiseComputeEx<RealMulComplexFunctor<T>, DeviceContext, T>(
        context, &x, &y, -1, RealMulComplexFunctor<T>(), &ret);
    return ret;
  }

  framework::Tensor Div(const framework::Tensor& x,
                        const framework::Tensor& y) {
    framework::Tensor ret;
    if (x.type() != y.type()) {
      ret.mutable_data<T>(x.dims(), context.GetPlace());
      auto x_vector = EigenVector<T>::Flatten(x);
      auto y_vector = EigenVector<ValueType>::Flatten(y);
      auto out_vector = EigenVector<T>::Flatten(ret);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      out_vector.device(place) = x_vector / y_vector;
    } else {
      std::vector<int> out_shape = GetBroadcastShape({&x, &y});
      ret.Resize(phi::make_ddim(out_shape));
      ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
          context, &x, &y, -1, DivFunctor<T>(), &ret);
    }
    return ret;
  }
  framework::Tensor Add(const framework::Tensor& x,
                        const framework::Tensor& y) {
    // element wise add, support numpy broadcast.
    framework::Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(phi::make_ddim(out_shape));
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
        context, &x, &y, -1, AddFunctor<T>(), &ret);
    return ret;
  }
  framework::Tensor Mul(const framework::Tensor& x,
                        const framework::Tensor& y) {
    framework::Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(phi::make_ddim(out_shape));
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
        context, &x, &y, -1, MulFunctor<T>(), &ret);
    return ret;
  }

  framework::Tensor ReduceSum(const framework::Tensor& x,
                              std::vector<int> out_dim) {
    framework::AttributeMap attrs;
    attrs["dim"] = std::vector<int>{-1};
    NameInTensorMap inputs({{"X", {&x}}});
    return CreateOpRunAndReturnTensor("reduce_sum", inputs, attrs, out_dim);
  }

  framework::Tensor ReduceMax(const framework::Tensor& x,
                              std::vector<int> out_dim) {
    framework::AttributeMap attrs;
    attrs["dim"] = std::vector<int>{-1};
    NameInTensorMap inputs({{"X", {&x}}});
    return CreateOpRunAndReturnTensor("reduce_max", inputs, attrs, out_dim);
  }
  // Support float and complex type subtraction，the default is T type
  template <typename InT = T>
  framework::Tensor Sub(const framework::Tensor& x,
                        const framework::Tensor& y) {
    framework::Tensor ret;
    std::vector<int> out_shape = GetBroadcastShape({&x, &y});
    ret.Resize(phi::make_ddim(out_shape));
    if (platform::is_gpu_place(context.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      // For GPU, there is no need to define XxxInverseFunctor and call
      // ElementwiseComputeEx in two branches.
      ElementwiseComputeEx<SubFunctor<InT>, DeviceContext, InT>(
          context, &x, &y, -1, SubFunctor<InT>(), &ret);
#endif
    } else {
      if (x.dims().size() >= y.dims().size()) {
        ElementwiseComputeEx<SubFunctor<InT>, DeviceContext, InT>(
            context, &x, &y, -1, SubFunctor<InT>(), &ret);
      } else {
        // This is copyed from elementwise_sub, which means we
        // need reverse will xrank < yrank
        ElementwiseComputeEx<InverseSubFunctor<InT>, DeviceContext, InT>(
            context, &x, &y, -1, InverseSubFunctor<InT>(), &ret);
      }
    }
    return ret;
  }
  const framework::Tensor Unsqueeze(const framework::Tensor& x, int axis = 0) {
    // don't copy data, only change the dims
    framework::Tensor out;
    out.ShareDataWith(x);
    std::vector<int> out_shape = phi::vectorize<int>(x.dims());
    if (axis >= 0) {
      auto index = (out_shape.begin() + axis);
      out_shape.insert(index, 1);
    } else if (axis < 0) {
      auto index = (out_shape.end() + axis + 1);
      out_shape.insert(index, 1);
    }
    out.Resize(phi::make_ddim(out_shape));
    return out;
  }
  framework::Tensor Fill(std::vector<int> shape, float fill_value) {
    framework::Tensor ret;
    ret.Resize(phi::make_ddim(shape));
    ret.mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    phi::funcs::SetConstant<DeviceContext, T>()(dev_ctx, &ret, T(fill_value));
    return ret;
  }
  framework::Tensor Infinits(std::vector<int> shape) {
    auto value = static_cast<T>(std::numeric_limits<double>::infinity());
    return Fill(shape, value);
  }
  framework::Tensor Eye(int n) {
    auto output = Fill({n}, 1);
    auto ret = Diag(output);
    return ret;
  }
  framework::Tensor Slice(const framework::Tensor& x, std::vector<int> axes,
                          std::vector<int> starts, std::vector<int> ends) {
    framework::Tensor ret;
    std::vector<int> new_axes = axes;
    std::vector<int> out_shape = phi::vectorize<int>(x.dims());
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
    ret.Resize(phi::make_ddim(out_shape));
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

  framework::Tensor TrilTriu(const framework::Tensor& x, int diagonal,
                             bool lower) {
    framework::AttributeMap attrs;
    attrs["diagonal"] = diagonal;
    attrs["lower"] = lower;
    NameInTensorMap inputs({{"X", {&x}}});
    int x_rank = x.dims().size();
    PADDLE_ENFORCE_GE(x_rank, 2, platform::errors::InvalidArgument(
                                     "Rank must be at least 2."));
    std::vector<int> out_shape = phi::vectorize<int>(x.dims());
    return CreateOpRunAndReturnTensor("tril_triu", inputs, attrs, out_shape);
  }

  framework::Tensor TriangularSolve(const framework::Tensor& x,
                                    const framework::Tensor& y, bool upper,
                                    bool transpose, bool unitriangular) {
    framework::AttributeMap attrs;
    attrs["upper"] = upper;
    attrs["transpose"] = transpose;
    attrs["unitriangular"] = unitriangular;
    NameInTensorMap inputs({{"X", {&x}}, {"Y", {&y}}});
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    auto y_dims_n = y_dims.size();
    std::vector<int64_t> x_dims_vec = phi::vectorize<int64_t>(x_dims);
    std::vector<int64_t> y_dims_vec = phi::vectorize<int64_t>(y_dims);
    std::vector<int64_t> x_dims_vec_cut(x_dims_vec.begin(),
                                        x_dims_vec.end() - 2);
    std::vector<int64_t> y_dims_vec_cut(y_dims_vec.begin(),
                                        y_dims_vec.end() - 2);
    std::vector<int64_t> expand_batch_portion =
        get_broadcast_batch_portion(x_dims_vec_cut, y_dims_vec_cut);
    std::vector<int64_t> y_broadcast_dims({expand_batch_portion});
    y_broadcast_dims.insert(y_broadcast_dims.end(), {y_dims_vec[y_dims_n - 2],
                                                     y_dims_vec[y_dims_n - 1]});
    std::vector<int> out_shape(y_broadcast_dims.begin(),
                               y_broadcast_dims.end());
    return CreateOpRunAndReturnTensor("triangular_solve", inputs, attrs,
                                      out_shape);
  }

  framework::Tensor ConcatTwoTensors(const framework::Tensor& x,
                                     const framework::Tensor& y, int axis) {
    framework::AttributeMap attrs;
    attrs["axis"] = axis;
    std::vector<framework::DDim> inputs_dims({x.dims(), y.dims()});
    NameInTensorMap inputs({{"X", {&x, &y}}});
    size_t axis_ =
        ComputeAxisForConcatOp(static_cast<int64_t>(axis),
                               static_cast<int64_t>(inputs_dims[0].size()));
    framework::DDim out_dims =
        ComputeAndCheckShapeForConcatOp(true, inputs_dims, axis_);
    if (out_dims[axis_] < 0) {
      out_dims[axis_] = -1;
    }
    std::vector<int> out_shape = phi::vectorize<int>(out_dims);
    return CreateOpRunAndReturnTensor("concat", inputs, attrs, out_shape);
  }

  Tensor Conj(const Tensor& x) {
    Tensor out;
    auto* out_data = out.mutable_data<T>(x.dims(), context.GetPlace());
    auto* x_data = x.data<T>();
    auto for_range = GetForRange(x.numel());
    phi::funcs::ConjFunctor<T> functor(x_data, x.numel(), out_data);
    for_range(functor);
    return out;
  }

  Tensor Real(const Tensor& x) {
    Tensor out;
    auto numel = x.numel();
    auto* out_data = out.mutable_data<phi::dtype::Real<T>>(
        x.dims(), context.GetPlace(),
        static_cast<size_t>(numel * sizeof(phi::dtype::Real<T>)));
    auto* x_data = x.data<T>();
    auto for_range = GetForRange(numel);
    phi::funcs::RealFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
    return out;
  }

  Tensor DiagFill(const int m, const int n, const int num_lower_diags,
                  const int num_upper_diags, const Tensor& scale,
                  const Tensor& input) {
    Tensor out;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, input.numel());
    DiagAndFillFunctor<T, ValueType> diag_and_copy_functor(
        m, n, num_lower_diags, num_upper_diags, scale.data<ValueType>(),
        input.data<T>(), out.mutable_data<T>(input.dims(), input.place()));
    for_range(diag_and_copy_functor);
    return out;
  }

 private:
  const framework::ExecutionContext& context;
  phi::funcs::BlasT<DeviceContext, T> GetBlas() {
    return phi::funcs::GetBlas<DeviceContext, T>(context);
  }
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
  framework::Tensor CreateOpRunAndReturnTensor(
      const std::string& type, const NameInTensorMap& inputs,
      const framework::AttributeMap& attrs, std::vector<int> out_shape,
      NameOutTensor out_str = {"Out"}) {
    // varialble set dims must be LoDTensor / SelectedRowTensor
    framework::Scope& local_scope = context.scope().NewScope();
    framework::VariableNameMap op_outputs;
    for (auto out_name : out_str) {
      local_scope.Var("tmp_" + out_name)->GetMutable<framework::LoDTensor>();
      op_outputs[out_name].emplace_back("tmp_" + out_name);
    }
    auto out_var = local_scope.Var("tmp_Out");  // return the Out
    // create Out Tensor and allocat memory
    out_var->GetMutable<framework::LoDTensor>()->mutable_data<T>(
        phi::make_ddim(out_shape), context.GetPlace());
    // phi::make_ddim(out_shape)
    framework::VariableNameMap op_inputs;
    int counter = 0;
    for (auto item : inputs) {
      auto& tensors = item.second;
      std::vector<std::string> name_vector;
      for (auto each_tensor : tensors) {
        // create score variable and reset the tensor.
        std::string _name = "tmp" + std::to_string(counter++);
        auto in_var = local_scope.Var(_name);  // create
        framework::LoDTensor tmp_tns;
        tmp_tns.ShareDataWith(*each_tensor);  // tensor -> lodtensor
        (*in_var->GetMutable<framework::LoDTensor>()) =
            tmp_tns;  // initialize and set value
        name_vector.emplace_back(_name);
      }
      op_inputs[item.first] = name_vector;
    }

    auto op =
        framework::OpRegistry::CreateOp(type, op_inputs, op_outputs, attrs);
    op->Run(local_scope, context.GetPlace());
    framework::Tensor out;
    out.ShareDataWith(*(out_var->GetMutable<framework::LoDTensor>()));
    out.Resize(phi::make_ddim(out_shape));
    context.scope().DeleteScope(&local_scope);
    return out;
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
