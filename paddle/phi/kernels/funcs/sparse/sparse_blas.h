//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename DeviceContext>
class SparseBlas {
 public:
  explicit SparseBlas(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}

  template <typename T, typename TensorType>
  void SPMM(bool transa,
            bool transb,
            T alpha,
            const TensorType& mat_a,
            const phi::DenseTensor& mat_b,
            T beta,
            phi::DenseTensor* mat_out) const;

  template <typename T, typename TensorType>
  void SPMV(bool transa,
            T alpha,
            const TensorType& mat_a,
            const phi::DenseTensor& vec_x,
            T beta,
            phi::DenseTensor* vec_out) const;

  template <typename T, typename TensorType>
  void SDDMM(bool transa,
             bool transb,
             T alpha,
             const phi::DenseTensor& mat_a,
             const phi::DenseTensor& mat_b,
             T beta,
             TensorType* mat_out) const;

 private:
  const DeviceContext& dev_ctx_;
};

template <typename DeviceContext, typename T>
class SparseBlasT : private SparseBlas<DeviceContext> {
 public:
  using SparseBlas<DeviceContext>::SparseBlas;

  template <typename... ARGS>
  void SPMM(ARGS... args) const {
    Base()->template SPMM<T>(args...);
  }

  template <typename... ARGS>
  void SPMV(ARGS... args) const {
    Base()->template SPMV<T>(args...);
  }

  template <typename... ARGS>
  void SDDMM(ARGS... args) const {
    Base()->template SDDMM<T>(args...);
  }

 private:
  const SparseBlas<DeviceContext>* Base() const {
    return static_cast<const SparseBlas<DeviceContext>*>(this);
  }
};

template <typename DeviceContext, typename T>
inline SparseBlasT<DeviceContext, T> GetSparseBlas(
    const DeviceContext& dev_ctx) {
  return SparseBlasT<DeviceContext, T>(dev_ctx);
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11000
#include "paddle/phi/kernels/funcs/sparse/sparse_blas_impl.cu.h"
#endif
#if defined(PADDLE_WITH_HIP) && HIP_VERSION >= 402
#include "paddle/phi/kernels/funcs/sparse/sparse_blas_impl.hip.h"
#endif
