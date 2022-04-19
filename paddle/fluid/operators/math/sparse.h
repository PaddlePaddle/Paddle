//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext>
class Sparse {
 public:
  explicit Sparse(const DeviceContext& context) : context_(context) {}

  template <typename T>
  void nnz(const int M, const int N, const T* dense, int* nnz,
           int* nnzPerRowColumn) const;

  template <typename T>
  void DenseToSparseCoo(const int M, const int N, const T* dense, int64_t* rows,
                        int64_t* cols, T* values) const;

  template <typename T>
  void DenseToSparseCsr(const int M, const int N, const T* dense,
                        int64_t* crows, int64_t* cols, T* values) const;

  template <typename T>
  void SparseCooToDense(const int64_t M, const int64_t N, const int64_t nnz,
                        const int64_t* rows, const int64_t* cols,
                        const T* values, T* dense) const;
  template <typename T>
  void SparseCsrToDense(const int64_t M, const int64_t N, const int64_t nnz,
                        const int64_t* crows, const int64_t* cols,
                        const T* values, T* dense) const;

 private:
  const DeviceContext& context_;
};

template <typename DeviceContext, typename T>
class SparseT : private Sparse<DeviceContext> {
 public:
  using Sparse<DeviceContext>::Sparse;

  template <typename... ARGS>
  void nnz(ARGS... args) const {
    Base()->template nnz<T>(args...);
  }

  template <typename... ARGS>
  void DenseToSparseCoo(ARGS... args) const {
    Base()->template DenseToSparseCoo<T>(args...);
  }
  template <typename... ARGS>
  void DenseToSparseCsr(ARGS... args) const {
    Base()->template DenseToSparseCsr<T>(args...);
  }
  template <typename... ARGS>
  void SparseCooToDense(ARGS... args) const {
    Base()->template SparseCooToDense<T>(args...);
  }
  template <typename... ARGS>
  void SparseCsrToDense(ARGS... args) const {
    Base()->template SparseCsrToDense<T>(args...);
  }

 private:
  const Sparse<DeviceContext>* Base() const {
    return static_cast<const Sparse<DeviceContext>*>(this);
  }
};

template <typename DeviceContext, typename T>
inline SparseT<DeviceContext, T> GetSparse(
    const framework::ExecutionContext& exe_ctx) {
  return SparseT<DeviceContext, T>(
      exe_ctx.template device_context<DeviceContext>());
}

template <typename DeviceContext, typename T>
inline SparseT<DeviceContext, T> GetSparse(const DeviceContext& dev_ctx) {
  return SparseT<DeviceContext, T>(dev_ctx);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 11020
#include "paddle/fluid/operators/math/sparse_impl.cu.h"
#endif
#endif
