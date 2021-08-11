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
#include <Eigen/SVD>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
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
using Shape = std::vector<long int>;
using OpName = std::string;
using namespace Eigen;
using namespace std;
using namespace framework;

template <typename T>
void EigenSvd(const T* X, T* U, T* VH, T* S, int rows, int cols) {
  // 这里为什么用参数row=2, cols=2？？？
  BDCSVD<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>> svd(
      2, 2, Eigen::DecompositionOptions::ComputeThinU |
                Eigen::DecompositionOptions::ComputeThinV);
  T* input = const_cast<T*>(X);
  auto m = Map<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>>(input, rows, cols);
  svd.compute(m);
  Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> V_trans =
      svd.matrixV().transpose();
  memcpy(U, svd.matrixU().data(), svd.matrixU().size() * sizeof(T));
  memcpy(VH, V_trans.data(), V_trans.size() * sizeof(T));
  memcpy(S, svd.singularValues().data(),
         svd.singularValues().size() * sizeof(T));
}

template <typename T>
void BatchSvd(const T* X, T* U, T* VH, T* S, int rows, int cols, int batches) {
  int stride = rows * cols;
  int k = std::min(rows, cols);
  for (int i = 0; i < batches; ++i) {
    EigenSvd<T>(X + i * stride, U + i * k * rows, VH + i * k * cols, S + i * k,
                rows, cols);
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void EigenSvdOnlyS(const T* X, T* S, int rows, int cols) {
  // 这里为什么用参数row=2, cols=2？？？
  BDCSVD<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>> svd(
      2, 2, Eigen::DecompositionOptions::ComputeThinU |
                Eigen::DecompositionOptions::ComputeThinV);
  T* input = const_cast<T*>(X);
  auto m = Map<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>>(input, rows, cols);
  svd.compute(m);
  memcpy(S, svd.singularValues().data(),
         svd.singularValues().size() * sizeof(T));
}

template <typename T>
void BatchSvdOnlyS(const T* X, T* S, int rows, int cols, int batches) {
  int stride = rows * cols;
  int k = std::min(rows, cols);
  for (int i = 0; i < batches; ++i) {
    EigenSvdOnlyS<T>(X + i * stride, S + i * k, rows, cols);
  }
}

template <typename T>
void rank(T* S, int32_t* out, float tol, int k) {
  int rank = 0;
  for (int i = 0; i < k; i++) {
    if ((*(S + i)) > tol) {
      rank = rank + 1;
    }
  }
  *out = rank;
}

template <typename T>
void BatchRank(T* S, int32_t* out, float tol, int batches, int k) {
  // int stride = rows * cols;
  // int k = std::min(rows, cols);
  for (int i = 0; i < batches; i++) {
    rank<T>(S + i * k, out + i, tol, k);
  }
}

template <typename T>
void BatchRankUseSVD(const T* x_data, int32_t* out_data, float* tol,
                     int batches, int rows, int cols) {
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  // int k = std::min(rows, cols);

  BDCSVD<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>> svd;
  for (int i = 0; i < batches; ++i) {
    // compute SVD
    auto m = Map<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    svd.compute(m);
    auto res_s = svd.singularValues();
    // compute tol
    float tol_val = std::numeric_limits<float>::epsilon() *
                    std::max(rows, cols) * res_s.maxCoeff();
    if (tol) {
      tol_val = std::max(*tol, tol_val);
    }
    // compute rank
    int rank = 0;
    for (int j = 0; j < res_s.size(); j++) {
      if (res_s[j] > tol_val) {
        rank = rank + 1;
      }
    }
    *(out_data + i) = rank;
    VLOG(3) << "tol_val: " << tol_val << std::endl;
  }
}

template <typename T>
void BatchRankUseEigenvalues(const T* x_data, int32_t* out_data, float* tol,
                             int batches, int rows, int cols) {
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    int rank = 0;
    auto m = Map<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    // eigenvalues type is complex<float>, check real part
    auto eigenvalues = m.eigenvalues().real().cwiseAbs();
    float tol_val = std::numeric_limits<float>::epsilon() *
                    std::max(rows, cols) * eigenvalues.maxCoeff();
    if (tol) {
      tol_val = std::max(*tol, tol_val);
    }
    for (int j = 0; j < eigenvalues.size(); j++) {
      if (eigenvalues[j] > tol_val) {
        rank = rank + 1;
      }
    }
    *(out_data + i) = rank;
    VLOG(3) << "tol_val: " << tol_val << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

template <typename T>
struct PowFunctor {
  PowFunctor(const T* input, T* output, int64_t numel, float exp)
      : input_(input), output_(output), numel_(numel), exp(exp) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = pow(input_[idx], exp);
  }
  const T* input_;
  T* output_;
  int64_t numel_;
  float exp;
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

/*
class FakeExecutionContext {
 public:
  using NameMapper = std::map<std::string, void *> ;
  FakeExecutionContext(const ExecutionContext & ctx, NameMapper & map)
      : context(ctx), mapper(map){}
 public:
  template<typename T>
  const T * Input(std::string name) const{
    return reinterpret_cast<T*>(mapper[name]) ;
  }
  template<typename T>
  T * Output(std::string name) const{
    return reinterpret_cast<T*>(mapper[name]) ;
  }
  template<typename T>
  T Attr(std::string name) const{
    return (* reinterpret_cast<T*>(mapper[name])) ;
  }
  operator const framework::ExecutionContext& () const{
    return context ;
  }
 private:
  const framework::ExecutionContext & context ;
  NameMapper & mapper ;
};
*/

static Shape _get_broadcast_shape(InTensors ins) {
  // TODO check the operators and output
  auto x_dim = ins[0]->dims();
  auto y_dim = ins[1]->dims();
  Shape ret = (x_dim.size() > y_dim.size() ? framework::vectorize(x_dim)
                                           : framework::vectorize(y_dim));
  int rank = min(x_dim.size(), y_dim.size());
  for (int i = rank - 1; i >= 0; --i) {
    if (x_dim[i] == y_dim[i]) {
      ret[i] = x_dim[i];
      continue;
    }
    if (x_dim[i] == 1) {
      ret[i] = y_dim[i];
      continue;
    }
    if (y_dim[i] == 1) {
      ret[i] = x_dim[i];
      continue;
    }
    PADDLE_ENFORCE_EQ(0, 1, "Wrong Input Shape in add operator");
  }
  return ret;
}

template <typename DeviceContext, typename T>
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

  framework::Tensor pow(const framework::Tensor& x, float exp) {
    framework::Tensor out;
    auto for_range = GetForRange(x.numel());
    check_output(out);
    int numel = x.numel();
    PowFunctor<T> functor(x.data<T>(), out.mutable_data<T>(x.dims(), x.place()),
                          numel, exp);
    for_range(functor);
    return out;
  }
  /*
  void matmul(const framework::Tensor& mat_a, bool trans_a,
            const framework::Tensor& mat_b, bool trans_b,
            framework::Tensor* mat_out){
      auto blas = GetBlas() ;
      check_output(* mat_out) ;
      blas.MatMul(mat_a, trans_a, mat_b, trans_b, mat_out) ;
  }
  */
  framework::Tensor matmul(const framework::Tensor& mat_a,
                           const framework::Tensor& mat_b, bool trans_a = false,
                           bool trans_b = false) {
    framework::AttributeMap attrs;
    attrs["trans_x"] = trans_a;
    attrs["trans_y"] = trans_b;
    NameInTensorMap inputs({{"X", {&mat_a}}, {"Y", {&mat_b}}});
    auto a_dim = mat_a.dims();
    auto b_dim = mat_b.dims();
    Shape x_vec = framework::vectorize(a_dim);
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
    auto x_vec = framework::vectorize(x_dim);
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

  /*
      framework::Tensor add(const framework::Tensor & x, const framework::Tensor
     & y) {
          VLOG(2) << "INPUT" << std::endl ;
          InTensors ins({&x, &y}) ;
          VLOG(2) << "after ins" << std::endl ;
          return elementwise_op("add", ins) ;
      }
  */

  framework::Tensor diag(const framework::Tensor& x, int offset = 0,
                         int padding_value = 0) {
    framework::AttributeMap attrs;
    attrs["offset"] = offset;
    attrs["padding_value"] = padding_value;
    NameInTensorMap inputs({{"X", {&x}}});
    int x_rank = x.dims().size();
    Shape out_shape;
    if (x_rank == 2) {
      PADDLE_ENFORCE_EQ(x.dims()[0], x.dims()[1],
                        "if X is a Matrix, then X must be square");
      out_shape.push_back(x.dims()[0]);
    } else if (x_rank == 1) {
      out_shape.push_back(x.dims()[0]);
      out_shape.push_back(x.dims()[0]);
    } else {
      PADDLE_ENFORCE_EQ(0, 1, "Rank must less or equal than 2");
    }
    return _CreateOpRunAndReturnTensor("diag_v2", inputs, attrs, out_shape);
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
    Shape out_shape = framework::vectorize(x.dims());
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

  framework::Tensor zeros(Shape shape, proto::VarType::Type dtype,
                          T fill_value = T(0)) {
    framework::AttributeMap attrs;
    attrs["dtype"] = dtype;
    attrs["shape"] = shape;
    attrs["value"] = 0.0f;
    framework::Tensor value;
    T* value_data = value.mutable_data<T>({1}, context.GetPlace());
    value_data[0] = T(fill_value);

    NameInTensorMap inputs({{"ValueTensor", {&value}}});
    return _CreateOpRunAndReturnTensor("fill_constant", inputs, attrs, shape);
  }

  framework::Tensor infinits(Shape shape, proto::VarType::Type dtype) {
    framework::AttributeMap attrs;
    attrs["dtype"] = dtype;
    attrs["shape"] = shape;
    attrs["str_value"] = std::string("inf");
    NameInTensorMap inputs({});
    return _CreateOpRunAndReturnTensor("fill_constant", inputs, attrs, shape);
  }

  framework::Tensor eye(int n, proto::VarType::Type dtype) {
    auto output = zeros({n}, dtype, 1);
    return diag(output);
  }

  void print_matrix(const framework::Tensor& x, std::string name) {
    auto dims = x.dims();
    int rank = dims.size();
    int rows = dims[rank - 2];
    int cols = dims[rank - 1];
    std::string ret("\n");
    VLOG(2) << "Tensor: " << name << "\nDIM: " << x.dims() << std::endl;
    const T* x_data = x.data<T>();
    if (rank == 1) {
      for (int j = 0; j < cols; ++j) {
        ret += std::to_string(*(x_data + j)) + "\t";
      }
      ret += "\n";
    } else {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          ret += std::to_string(*(x_data + i * cols + j)) + "\t";
        }
        ret += "\n";
      }
    }
    VLOG(2) << ret << std::endl;
  }

 private:
  const framework::ExecutionContext& context;

  void check_output(framework::Tensor& output) {
    assert(output.IsInitialized() == true);
  }
  BlasT<DeviceContext, T> GetBlas() {
    return math::GetBlas<DeviceContext, T>(context);
  }
  platform::ForRange<DeviceContext> GetForRange(int numel) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    return platform::ForRange<DeviceContext>(dev_ctx, numel);
  }

  /*
      framework::Tensor elementwise_op(OpName name,
                                       InTensors op_args) {
          return ElementWiseWrapper<DeviceContext, T>::elementwise_op(name,
     op_args, context) ;
      }
  */

  framework::Tensor _CreateOpRunAndReturnTensor(
      const std::string& type, NameInTensorMap& inputs,
      const AttributeMap& attrs, Shape out_shape,
      NameOutTensor out_str = {"Out"}) {
    // varialble set dims must be LoDTensor / SelectedRowTensor
    Scope& local_scope = context.scope().NewScope();

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
    /*  print the temp variable.
    for (auto item : op_inputs) {
        VLOG(2) << item.first << ":  [" << item.second[0] << "]" << std::endl ;
    }
    */
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
}
}
}
