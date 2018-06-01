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

#include <glog/logging.h>
#include "paddle/function/EigenThreadDevice.h"

namespace paddle {

template <class T>
struct EigenBlasGemm {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, int>,
                           Eigen::Aligned>
      EigenMatrix;

  static void compute(const bool transA,
                      const bool transB,
                      const int M,
                      const int N,
                      const int K,
                      const T alpha,
                      const T* A,
                      const int lda,
                      const T* B,
                      const int ldb,
                      const T beta,
                      T* C,
                      const int ldc) {
    Eigen::array<int, 2> sizeA;
    if (transA) {
      sizeA[0] = K;
      sizeA[1] = M;
      CHECK_EQ(M, lda);
    } else {
      sizeA[0] = M;
      sizeA[1] = K;
      CHECK_EQ(K, lda);
    }
    Eigen::array<int, 2> sizeB;
    if (transB) {
      sizeB[0] = N;
      sizeB[1] = K;
      CHECK_EQ(K, ldb);
    } else {
      sizeB[0] = K;
      sizeB[1] = N;
      CHECK_EQ(N, ldb);
    }
    Eigen::array<int, 2> sizeC = {{M, ldc}};
    Eigen::array<int, 2> offsetC = {{0, 0}};
    Eigen::array<int, 2> extentC = {{M, N}};

    const EigenMatrix a(const_cast<T*>(A), sizeA);
    const EigenMatrix b(const_cast<T*>(B), sizeB);
    EigenMatrix c(C, sizeC);

    typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    dims[0].first = transA ? 0 : 1;
    dims[0].second = transB ? 1 : 0;

    auto* device = EigenDeviceWarpper::device();
    if (N == ldc) {
      if (alpha == T(1) && beta == T(0)) {
        c.device(*device) = a.contract(b, dims);
      } else if (alpha == T(1) && beta == T(1)) {
        c.device(*device) += a.contract(b, dims);
      } else {
        c.device(*device) = alpha * a.contract(b, dims) + beta * c;
      }
    } else {
      if (alpha == T(1) && beta == T(0)) {
        c.slice(offsetC, extentC).device(*device) = a.contract(b, dims);
      } else if (alpha == T(1) && beta == T(1)) {
        c.slice(offsetC, extentC).device(*device) += a.contract(b, dims);
      } else {
        c.slice(offsetC, extentC).device(*device) =
            alpha * a.contract(b, dims) + beta * c.slice(offsetC, extentC);
      }
    }
    EigenDeviceWarpper::free_device(device);
  }
};

#ifdef PADDLE_TYPE_DOUBLE
template struct EigenBlasGemm<double>;
#else
template struct EigenBlasGemm<float>;
#endif

}  // namespace paddle
