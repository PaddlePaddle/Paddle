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

#include <vector>
#include <complex>
#include "Eigen/Dense"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T> 
struct PaddleComplex{
  using Type = paddle::platform::complex<T>;
};
template <> 
struct PaddleComplex<paddle::platform::complex<float>>{
  using Type = paddle::platform::complex<float>;
};
template <> 
struct PaddleComplex<paddle::platform::complex<double>>{
  using Type = paddle::platform::complex<double>;
};

template <typename T> 
struct StdComplex{
  using Type = std::complex<T>;
};
template <> 
struct StdComplex<paddle::platform::complex<float>>{
  using Type = std::complex<float>;
};
template <> 
struct StdComplex<paddle::platform::complex<double>>{
  using Type = std::complex<double>;
};

template <typename T> 
using PaddleCType = typename PaddleComplex<T>::Type;
template <typename T> 
using StdCType = typename StdComplex<T>::Type;
template <typename T> 
using EigenMatrixPaddle = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> 
using EigenVectorPaddle = Eigen::Matrix<PaddleCType<T>, Eigen::Dynamic, 1>;
template <typename T> 
using EigenMatrixStd = Eigen::Matrix<StdCType<T>, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> 
using EigenVectorStd = Eigen::Matrix<StdCType<T>, Eigen::Dynamic, 1>;

static void SpiltBatchSquareMatrix(const Tensor* input, std::vector<Tensor>& output){
  DDim input_dims = input -> dims();
  int last_dim = input_dims.size() - 1;
  int n_dim = input_dims[last_dim];

  DDim flattened_input_dims, flattened_output_dims; 
  if(input_dims.size() > 2){
    flattened_input_dims = flatten_to_3d(input_dims, last_dim - 1, last_dim);
  }
  else{
    flattened_input_dims = framework::make_ddim({1, n_dim, n_dim});
  }

  Tensor flattened_input;
  flattened_input.ShareDataWith(*input);
  flattened_input.Resize(flattened_input_dims); 
  output = flattened_input.Split(1, 0); 
}

template <typename DeviceContext, typename T>
class EigvalsKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *input = ctx.Input<Tensor>("X");
    Tensor *output = ctx.Output<Tensor>("Out");
    
    auto input_type = input -> type();
    auto output_type = framework::IsComplexType(input_type) ? 
      input_type : framework::ToComplexType(input_type);
    output -> mutable_data(ctx.GetPlace(), output_type);

    std::vector<Tensor> input_matrices;
    SpiltBatchSquareMatrix(input, /*->*/ input_matrices);
    
    int n_dim = input_matrices[0].dims()[1];
    int n_batch = input_matrices.size();

    DDim output_dims = output->dims();
    output -> Resize(framework::make_ddim({n_batch, n_dim}));
    std::vector<Tensor> output_vectors = output->Split(1, 0);

    Eigen::Map<EigenMatrixPaddle<T>> input_emp(NULL, n_dim, n_dim);
    Eigen::Map<EigenVectorPaddle<T>> output_evp(NULL, n_dim);
    EigenMatrixStd<T> input_ems;
    EigenVectorStd<T> output_evs;
    
    for(int i = 0; i < n_batch; ++i){
      new (&input_emp) Eigen::Map<EigenMatrixPaddle<T>>(
        input_matrices[i].data<T>(), n_dim, n_dim);
      new (&output_evp) Eigen::Map<EigenVectorPaddle<T>>(
        output_vectors[i].data<PaddleCType<T>>(), n_dim);
      input_ems = input_emp.template cast<StdCType<T>>();
      output_evs = input_ems.eigenvalues();
      output_evp = output_evs.template cast<PaddleCType<T>>();
    }
    output -> Resize(output_dims);
  }
};


template<typename T>
inline void CastToPaddleType(
  EigenMatrixStd<T>& input, 
  Eigen::Map<EigenMatrixPaddle<T>>& output){
  output = input.template cast<T>();
}
template<>
inline void CastToPaddleType(
  EigenMatrixStd<float>& input, 
  Eigen::Map<EigenMatrixPaddle<float>>& output){
  output = input.real();
}
template<>
inline void CastToPaddleType(
  EigenMatrixStd<double>& input, 
  Eigen::Map<EigenMatrixPaddle<double>>& output){
  output = input.real();
}

template <typename DeviceContext, typename T>
class EigvalsGradKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("X");
    const Tensor* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad -> mutable_data(ctx.GetPlace(), input -> type());

    Tensor output;
    output.Resize(output_grad -> dims());
    output.mutable_data(ctx.GetPlace(), output_grad -> type());

    std::vector<Tensor> input_matrices, input_grad_matrices;
    SpiltBatchSquareMatrix(input, /*->*/ input_matrices);
    SpiltBatchSquareMatrix(input_grad, /*->*/ input_grad_matrices);

    int n_dim = input_matrices[0].dims()[1];
    int n_batch = input_matrices.size();

    Tensor flattened_output_grad;
    flattened_output_grad.ShareDataWith(*output_grad);
    flattened_output_grad.Resize(framework::make_ddim({n_batch, n_dim}));
    std::vector<Tensor> output_grad_vectors = flattened_output_grad.Split(1, 0);

    /*
    input_emp -> input_ems -> v_ems -> vh_ems 
    output_grad_evp -> output_grad_evs
    output_grad_evs + vh_ems -> input_grad_ems -> input_grad_emp
    */
    Eigen::Map<EigenMatrixPaddle<T>> input_emp(NULL, n_dim, n_dim);
    Eigen::Map<EigenMatrixPaddle<T>> input_grad_emp(NULL, n_dim, n_dim);
    Eigen::Map<EigenVectorPaddle<T>> output_grad_evp(NULL, n_dim);
    EigenMatrixStd<T> input_ems;
    EigenVectorStd<T> output_grad_evs;
    EigenMatrixStd<T> v_ems;
    EigenMatrixStd<T> input_grad_ems;

    for(std::vector<Tensor>::size_type i = 0; i < input_matrices.size(); ++i){
      new (&input_emp) Eigen::Map<EigenMatrixPaddle<T>>(
        input_matrices[i].data<T>(), n_dim, n_dim);
      new (&input_grad_emp) Eigen::Map<EigenMatrixPaddle<T>>(
        input_grad_matrices[i].data<T>(), n_dim, n_dim);
      new (&output_grad_evp) Eigen::Map<EigenVectorPaddle<T>>(
        output_grad_vectors[i].data<PaddleCType<T>>(), n_dim);

      /**
       * Let the input square matrix 
       *   A = VLV^{-1}, 
       * the gradient of A for eigenvalues L is
       *   A_grad = V^{-H} L_grad V^H.
       * See Eq. 4.77 in https://arxiv.org/pdf/1701.00392.pdf
       */
      input_ems = input_emp.template cast<StdCType<T>>();
      output_grad_evs = output_grad_evp.template cast<StdCType<T>>();
VLOG(4) << "input_ems:\n" << input_ems;
VLOG(4) << "output_grad_evs:\n" << output_grad_evs;
      Eigen::ComplexEigenSolver<EigenMatrixStd<T>> es(input_ems);
      v_ems = es.eigenvectors();
VLOG(4) << "eigenvalues:\n" << es.eigenvalues();
VLOG(4) << "v_ems:\n" << v_ems;
      v_ems.adjointInPlace();
VLOG(4) << "vh_ems:\n" << v_ems;
      input_grad_ems = v_ems.colPivHouseholderQr().solve(output_grad_evs.asDiagonal() * v_ems);
VLOG(4) << "input_grad_ems:\n" << input_grad_ems;
      CastToPaddleType<T>(input_grad_ems, /*->*/ input_grad_emp);
VLOG(4) << "input_grad_emp:\n" << input_grad_emp;
    }


  }
};
}  // namespace operators
}  // namespace paddle
