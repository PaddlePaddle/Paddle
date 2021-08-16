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
#include <Eigen/SVD>
#include <cstdarg>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T>
void BatchRankUseSVD(const T* x_data, int32_t* out_data, float tol, int batches,
                     int rows, int cols) {
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  // int k = std::min(rows, cols);

  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd;
  for (int i = 0; i < batches; ++i) {
    // compute SVD
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    svd.compute(m);
    auto res_s = svd.singularValues();
    // compute tol
    float tol_val = std::numeric_limits<float>::epsilon() *
                    std::max(rows, cols) * res_s.maxCoeff();
    tol_val = std::max(tol, tol_val);
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
void BatchRankUseEigenvalues(const T* x_data, int32_t* out_data, float tol,
                             int batches, int rows, int cols) {
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    // compute eigenvalues
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    // eigenvalues type is complex<float>, check real part
    auto eigenvalues = m.eigenvalues().real().cwiseAbs();
    float tol_val = std::numeric_limits<float>::epsilon() *
                    std::max(rows, cols) * eigenvalues.maxCoeff();
    tol_val = std::max(tol, tol_val);
    // compute rank
    int rank = 0;
    for (int j = 0; j < eigenvalues.size(); j++) {
      if (eigenvalues[j] > tol_val) {
        rank = rank + 1;
      }
    }
    *(out_data + i) = rank;
    VLOG(3) << "tol_val: " << tol_val << std::endl;
  }
}

template <typename T>
class MatrixRankCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get input/output
    const Tensor* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    Tensor* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<int32_t>(context.GetPlace());
    float tol = context.Attr<float>("tol");
    bool hermitian = context.Attr<bool>("hermitian");

    // get shape
    auto x_dims = x->dims();
    int rows = x_dims[x_dims.size() - 2];
    int cols = x_dims[x_dims.size() - 1];
    auto numel = x->numel();
    // int k = std::min(rows, cols);
    int batches = numel / (rows * cols);

    // compute
    if (hermitian) {
      PADDLE_ENFORCE_EQ(rows, cols,
                        platform::errors::InvalidArgument(
                            "if hermitian == true, rows == cols for matrix"));
      BatchRankUseEigenvalues<T>(x_data, out_data, tol, batches, rows, cols);
    } else {
      BatchRankUseSVD<T>(x_data, out_data, tol, batches, rows, cols);

      // Tensor s;
      // auto* s_data = s.mutable_data<T>( context.GetPlace(), size_t(batches *
      // std::min(rows, cols) * sizeof(T)) );
      /*SVD Use the Eigen Library*/
      // math::BatchSvdOnlyS<T>(x_data, s_data, rows, cols, batches);
      // coumpute rank
      // for (int i = 0; i < batches; i++) {
      //   math::BatchRank<T>(s_out, out_data, *tol, batches, k);
      // }
    }

    // 广播可以使用expand算子/broadcast_tensors算子/broadcast_shape、broadcast_tensors
  }
};
}  // namespace operators
}  // namespace paddle
