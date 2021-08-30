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
#include <assert.h>
#include <Eigen/Dense>
#include "Eigen/Core"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/math/math_function.h"
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

template <typename ValueType>
void BatchEigenvalues(ValueType* x_data, ValueType* eigenvalues_data,
                      ValueType* eigenvectors_data, int batches, int rows,
                      int cols, int k) {
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>>(x_data + i * stride,
                                                        rows, cols);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<
        ValueType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    PADDLE_ENFORCE_EQ(eigen_solver.info(), Eigen::Success,
                      platform::errors::InvalidArgument(
                          "Self Adjoint Eigen decomposition was"
                          "not successful. The "
                          "%d-th input matrice "
                          "might not be not be positive definite.",
                          i));
    auto eigenvalues = eigen_solver.eigenvalues().transpose();
    auto eigenvectors = eigen_solver.eigenvectors();
    memcpy(eigenvalues_data + i * rows, eigenvalues.data(),
           rows * sizeof(ValueType));
    memcpy(eigenvectors_data + i * stride, eigenvectors.data(),
           eigenvectors.size() * sizeof(ValueType));
  }
}

template <typename T, typename ValueType>
void BatchComplexValues(T* x_data, ValueType* eigenvalues_data,
                        T* eigenvectors_data, int batches, int rows, int cols,
                        int k) {
  std::complex<ValueType>* input =
      reinterpret_cast<std::complex<ValueType>*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<Eigen::Matrix<std::complex<ValueType>, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<std::complex<ValueType>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>>
        eigen_solver(m);
    PADDLE_ENFORCE_EQ(eigen_solver.info(), Eigen::Success,
                      platform::errors::InvalidArgument(
                          "Self Adjoint Eigen decomposition was"
                          "not successful. The "
                          "%d-th input matrice "
                          "might not be not be positive definite.",
                          i));

    auto eigenvalues = eigen_solver.eigenvalues().transpose();
    auto eigenvectors = eigen_solver.eigenvectors();
    memcpy(eigenvalues_data + i * rows, eigenvalues.data(),
           rows * sizeof(ValueType));

    memcpy(eigenvectors_data + i * stride, eigenvectors.data(),
           eigenvectors.size() * sizeof(T));
  }
}

template <typename T>
struct TransposeFunctor {
  TransposeFunctor(const T* input, T* output, int64_t numel, int64_t rows,
                   int64_t cols)
      : input_(input), output_(output), numel_(numel), rows(rows), cols(cols) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    int64_t batch_num = idx % (rows * cols) * (rows * cols);
    int64_t out_idx =
        (idx - batch_num) % cols * rows + (idx - batch_num) / cols;
    output_[out_idx] = input_[idx];
  }
  const T* input_;
  T* output_;
  int64_t numel_;
  int64_t rows;
  int64_t cols;
};

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

/*
template <typename T>
struct DiagFillFunctor {
  DiagFillFunctor(const T input, T * output, int64_t diag_number)
      : input_(input), output_(output), numel_(diag_number) {}
  HOSTDEVICE void operator()(int64_t idx) const {
    int64_t outer_batch_id = idx  / (numel_) * (numel_ * numel_) ;
    int64_t inner_batch_id = (idx  %  numel_) * (numel_ + 1) ;
    int64_t out_idx = outer_batch_id + inner_batch_id ;
    output_[out_idx] = input_;
  }
  const T input_;
  T* output_;
  int64_t numel_;
  float exp ;
};
*/

static Shape _get_broadcast_shape(InTensors ins) {
  // TODO(xiongkun03) check the operators and output
  auto x_dim = ins[0]->dims();
  auto y_dim = ins[1]->dims();
  Shape ret = (x_dim.size() > y_dim.size() ? framework::vectorize<int>(x_dim)
                                           : framework::vectorize<int>(y_dim));
  int rank = std::min(x_dim.size(), y_dim.size());
  int rx = x_dim.size();
  int ry = y_dim.size();
  int rr = ret.size();
  for (int i = 1; i <= rank; ++i) {
    if (x_dim[rx - i] == y_dim[ry - i]) {
      ret[rr - i] = x_dim[rx - i];
      continue;
    }
    if (x_dim[rx - i] == 1) {
      ret[rr - i] = y_dim[ry - i];
      continue;
    }
    if (y_dim[ry - i] == 1) {
      ret[rr - i] = x_dim[rx - i];
      continue;
    }
    PADDLE_ENFORCE_EQ(
        0, 1,
        platform::errors::InvalidArgument(
            "Wrong Input Shape in broadcast operator: "
            "Input(X)'s shape must follow the broadcast rule with Input(Y)'s "
            "shape, but received [%s] (X) vs [%s] (Y).",
            x_dim, y_dim));
  }
  return ret;
}

template <typename DeviceContext, typename T, typename ValueType>
struct DeviceIndependenceTensorOperations {
  // 1. Device Indenpendence, Kernel Reuse
  // 2. Tensor is always the input and output
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

  Tensor diag_copy(const int m, const int n, const int num_lower_diags,
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

  // void copy(const Tensor &input, Tensor& output){
  //   auto& dev_ctx = context.template device_context<DeviceContext>();
  //   paddle::framework::TensorCopy(
  //       input, input->place(), dev_ctx,
  //       output);  // copy input data to temp data

  // }
  /*
  void matmul(const framework::Tensor& mat_a, bool trans_a,
            const framework::Tensor& mat_b, bool trans_b,
            framework::Tensor* mat_out){
      auto blas = GetBlas() ;
      check_output(* mat_out) ;
      blas.MatMul(mat_a, trans_a, mat_b, trans_b, mat_out) ;
  }
  */
  // upper
  // Tensor triu_(const Tensor& x) {
  //   Shape out_shape = framework::vectorize<int>(x.dims());
  //   framework::AttributeMap attrs;
  //   attrs["diagonal"] = 0;
  //   attrs["lower"] = false;
  //   NameInTensorMap inputs({{"X", {&x}}});
  //   return _CreateOpRunAndReturnTensor("tril_triu", inputs, attrs,
  //   out_shape);
  // }

  // // lower
  // Tensor tril_(const Tensor& x) {
  //   Shape out_shape = framework::vectorize<int>(x.dims());
  //   framework::AttributeMap attrs;
  //   attrs["diagonal"] = 0;
  //   attrs["lower"] = true;
  //   NameInTensorMap inputs({{"X", {&x}}});
  //   return _CreateOpRunAndReturnTensor("tril_triu", inputs, attrs,
  //   out_shape);
  // }

  framework::Tensor matmul(const framework::Tensor& mat_a,
                           const framework::Tensor& mat_b, bool trans_a = false,
                           bool trans_b = false) {
    framework::AttributeMap attrs;
    attrs["trans_x"] = trans_a;
    attrs["trans_y"] = trans_b;
    NameInTensorMap inputs({{"X", {&mat_a}}, {"Y", {&mat_b}}});
    auto a_dim = mat_a.dims();
    auto b_dim = mat_b.dims();
    Shape x_vec = framework::vectorize<int>(a_dim);
    x_vec[x_vec.size() - 2] = a_dim[a_dim.size() - (trans_a ? 1 : 2)];
    x_vec[x_vec.size() - 1] = b_dim[b_dim.size() - (trans_b ? 2 : 1)];
    return _CreateOpRunAndReturnTensor("matmul_v2", inputs, attrs, x_vec);
  }
  // transpose the last two dimision
  framework::Tensor transpose(const framework::Tensor& x) {
    // PADDLE_ENFORCE_EQ(0, 1, "The Function Still have bugs, use
    // matmul(transpose=True)") ;
    framework::Tensor out;
    auto x_dim = x.dims();
    auto x_vec = framework::vectorize<int>(x_dim);
    int rank = x_vec.size();
    std::swap(x_vec[rank - 1], x_vec[rank - 2]);
    Shape out_shape = x_vec;
    std::vector<int> axis(rank);
    for (int i = 0; i < rank; ++i) {
      axis[i] = i;
    }
    std::swap(axis[rank - 1], axis[rank - 2]);
    framework::AttributeMap attrs;
    attrs["axis"] = axis;
    NameInTensorMap inputs({{"X", {&x}}});
    return _CreateOpRunAndReturnTensor("transpose2", inputs, attrs, out_shape,
                                       {"Out", "XShape"});
  }

  framework::Tensor conj_(const framework::Tensor& x) {
    // InTensors ins({&x});
    Shape out_shape = framework::vectorize<int>(x.dims());
    framework::AttributeMap attrs;
    NameInTensorMap inputs({{"X", {&x}}});
    return _CreateOpRunAndReturnTensor("conj", inputs, attrs, out_shape);
  }

  framework::Tensor add(const framework::Tensor& x,
                        const framework::Tensor& y) {
    InTensors ins({&x, &y});
    framework::AttributeMap attrs;
    attrs["axis"] = -1;
    Shape out_shape = _get_broadcast_shape({&x, &y});
    NameInTensorMap inputs({{"X", {&x}}, {"Y", {&y}}});
    return _CreateOpRunAndReturnTensor("elementwise_add", inputs, attrs,
                                       out_shape);
  }

  framework::Tensor mul(const framework::Tensor& x,
                        const framework::Tensor& y) {
    InTensors ins({&x, &y});
    framework::AttributeMap attrs;
    attrs["axis"] = -1;
    Shape out_shape = _get_broadcast_shape({&x, &y});
    NameInTensorMap inputs({{"X", {&x}}, {"Y", {&y}}});
    return _CreateOpRunAndReturnTensor("elementwise_mul", inputs, attrs,
                                       out_shape);
  }

  framework::Tensor div(const framework::Tensor& x,
                        const framework::Tensor& y) {
    InTensors ins({&x, &y});
    framework::AttributeMap attrs;
    attrs["axis"] = -1;
    Shape out_shape = _get_broadcast_shape({&x, &y});
    NameInTensorMap inputs({{"X", {&x}}, {"Y", {&y}}});
    return _CreateOpRunAndReturnTensor("elementwise_div", inputs, attrs,
                                       out_shape);
  }

  framework::Tensor sub(const framework::Tensor& x,
                        const framework::Tensor& y) {
    InTensors ins({&x, &y});
    framework::AttributeMap attrs;
    attrs["axis"] = -1;
    Shape out_shape = _get_broadcast_shape({&x, &y});
    NameInTensorMap inputs({{"X", {&x}}, {"Y", {&y}}});
    return _CreateOpRunAndReturnTensor("elementwise_sub", inputs, attrs,
                                       out_shape);
  }

  const framework::Tensor unsqueeze(const framework::Tensor& x, int axis = 0) {
    // don't copy data, only change the dims
    framework::Tensor out;
    out.ShareDataWith(x);
    Shape out_shape = framework::vectorize<int>(x.dims());
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

  framework::Tensor zeros(Shape shape, framework::proto::VarType::Type dtype,
                          float fill_value) {
    framework::AttributeMap attrs;
    attrs["dtype"] = dtype;
    attrs["shape"] = shape;
    attrs["value"] = fill_value;
    NameInTensorMap inputs({});
    return _CreateOpRunAndReturnTensor("fill_constant", inputs, attrs, shape);
  }

 private:
  const framework::ExecutionContext& context;

  void check_output(const framework::Tensor& output) {
    assert(output.IsInitialized() == true);
  }
  BlasT<DeviceContext, T> GetBlas() {
    return math::GetBlas<DeviceContext, T>(context);
  }
  platform::ForRange<DeviceContext> GetForRange(int numel) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    return platform::ForRange<DeviceContext>(dev_ctx, numel);
  }

  framework::Tensor _CreateOpRunAndReturnTensor(
      const std::string& type, const NameInTensorMap& inputs,
      const framework::AttributeMap& attrs, Shape out_shape,
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
        framework::make_ddim(out_shape), context.GetPlace());
    // framework::make_ddim(out_shape)
    framework::VariableNameMap op_inputs;
    int counter = 0;
    for (auto item : inputs) {
      std::string name = item.first;
      auto vec = item.second;
      std::vector<std::string> name_vector;
      for (auto vec_i : vec) {
        // create score variable and reset the tensor.
        std::string _name = "tmp" + std::to_string(counter++);
        auto in_var = local_scope.Var(_name);  // create
        framework::LoDTensor tmp_tns;
        tmp_tns.ShareDataWith(*vec_i);  // tensor -> lodtensor
        (*in_var->GetMutable<framework::LoDTensor>()) =
            tmp_tns;  // initialize and set value
        name_vector.emplace_back(_name);
      }
      op_inputs[name] = name_vector;
    }
    auto op =
        framework::OpRegistry::CreateOp(type, op_inputs, op_outputs, attrs);
    op->Run(local_scope, context.GetPlace());
    framework::Tensor out;
    out.ShareDataWith(*(out_var->GetMutable<framework::LoDTensor>()));
    out.Resize(framework::make_ddim(out_shape));
    context.scope().DeleteScope(&local_scope);
    return out;
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
