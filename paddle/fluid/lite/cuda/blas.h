// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <cublasXt.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <library_types.h>
#include "paddle/fluid/lite/cuda/cuda_utils.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace cuda {

#define CUBLAS_CHECK(xxx) CHECK_EQ((xxx), CUBLAS_STATUS_SUCCESS);

/*
 * Some basic methods.
 */
struct BlasBase {
  BlasBase() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~BlasBase() { CUBLAS_CHECK(cublasDestroy(handle_)); }

  void SetStream(cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
  }

  cudaStream_t GetStream() const {
    cudaStream_t stream;
    CUBLAS_CHECK(cublasGetStream_v2(handle_, &stream));
    return stream;
  }

  int GetVersion() const {
    int version{};
    CUBLAS_CHECK(cublasGetVersion_v2(handle_, &version));
    return version;
  }

  cublasHandle_t& handle() const { return handle_; }

 protected:
  // Not thread-safe, should created for each thread.
  // According to cublas doc.
  mutable cublasHandle_t handle_;
};

// T: Scalar type.
template <typename T>
class Blas : public lite::cuda::BlasBase {
 public:
  void sgemm(cublasOperation_t transa, cublasOperation_t transb,  //
             int m, int n, int k,                                 //
             const T* alpha,                                      //
             const T* A, int lda,                                 //
             const T* B, int ldb,                                 //
             const T* beta,                                       //
             T* C, int ldc) const {
    LITE_UNIMPLEMENTED;
  }
};

}  // namespace cuda
}  // namespace lite
}  // namespace paddle
