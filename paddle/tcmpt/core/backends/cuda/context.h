/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda_runtime.h>

#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/platform/gpu_info.h"

#include "paddle/fluid/platform/cuda_helper.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

#include "paddle/tcmpt/core/allocator.h"

namespace paddle {
namespace tcmpt {

class CUDAContext {
 public:
  explicit CUDAContext(const platform::Place& place) : place_(place) {}

  const platform::Place& place() const noexcept { return place_; }

  void SetAllocator(Allocator* allocator) { allocator_ = allocator; }

  Allocator* allocator() const noexcept { return allocator_; }

  void SetStream(cudaStream_t* stream) { stream_ = stream; }

  cudaStream_t* stream() const noexcept { return stream_; }

  void SetDnnHandle(cudnnHandle_t* handle) { dnn_handle_ = handle; }

  cudnnHandle_t* dnn_handle() const noexcept { return dnn_handle_; }

  void SetBlasHandle(cublasHandle_t* handle) { blas_handle_ = handle; }

  cublasHandle_t* blas_handle() const noexcept { return blas_handle_; }

 private:
  platform::Place place_;
  Allocator* allocator_{nullptr};
  cudaStream_t* stream_{nullptr};
  cudnnHandle_t* dnn_handle_{nullptr};
  cublasHandle_t* blas_handle_{nullptr};
  // cusolverDnHandle_t* solver_dn_handle_;
};

}  // namespace tcmpt
}  // namespace paddle
