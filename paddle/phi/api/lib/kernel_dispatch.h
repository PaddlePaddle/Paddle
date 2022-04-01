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

#include <limits>
#include <string>
#include <utility>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/backend_set.h"
#include "paddle/phi/api/lib/data_type_set.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

// TODO(chenweihang): split Key, Kernel, Factory into diff files
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace experimental {

namespace detail {
BackendSet GetTensorBackendSet(const phi::TensorBase& t);
std::size_t CountLeadingZeros(uint64_t val);
}  // namespace detail

phi::DeviceContext* GetDeviceContextByBackend(phi::Backend backend);

enum class KernelType {
  DENSE_TENSOR_KENREL,   // kernel for DenseTensor
  SELECTED_ROWS_KENREL,  // kernel for SelectedRows
  SPARSE_COO_KERNEL,     // kernel for SparseCooTensor
  SPARSE_CSR_KERNEL      // kernel for SparseCsrTensor
};

// TODO(chenweihang): support DataLayout and DataType selected
struct KernelKeySet {
  BackendSet backend_set{Backend::UNDEFINED};
  DataLayout layout{DataLayout::UNDEFINED};
  DataType dtype{DataType::UNDEFINED};

  // TODO(chenweihang): iterate all kernelkey for kernel selection
  phi::KernelKey GetHighestPriorityKernelKey() {
    return phi::KernelKey(static_cast<Backend>(64 - detail::CountLeadingZeros(
                                                        backend_set.bitset())),
                          layout,
                          dtype);
  }
};

namespace detail {

template <typename Functor>
struct ArgsIterator {
  template <typename... Args>
  inline Functor& apply() {
    return self();
  }

  template <typename T, typename... Args>
  inline Functor& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  constexpr bool short_circuit() const { return false; }

 private:
  inline Functor& self() { return *static_cast<Functor*>(this); }
};

struct KernelKeyParser : ArgsIterator<KernelKeyParser> {
  KernelKeySet key_set;
  // this dtype_set is used for cache multi-inputs dtype and used for
  // data_promote
  DataTypeSet dtype_set{DataType::UNDEFINED};

  // TODO(chenweihang): deal with multiple diff input Tensors
  // TODO(chenweihang): add global device guard method to set backend
  void operator()(const Tensor& x) {
    const phi::TensorBase& tensor = *x.impl();
    key_set.backend_set =
        key_set.backend_set | detail::GetTensorBackendSet(tensor);
    // TODO(chenweihang): select multi layout and dtype
    key_set.layout = tensor.layout();
    key_set.dtype = tensor.dtype();
    dtype_set = dtype_set | DataTypeSet(key_set.dtype);
    auto promote_result = PromoteTypes(dtype_set);
    if (promote_result != DataType::UNDEFINED) {
      key_set.dtype = promote_result;
    }
  }

  void operator()(const std::vector<Tensor>& x) {
    const phi::TensorBase& tensor = *x.at(0).impl();
    key_set.backend_set =
        key_set.backend_set | detail::GetTensorBackendSet(tensor);
    // TODO(chenweihang): select multi layout and dtype
    key_set.layout = tensor.layout();
    key_set.dtype = tensor.dtype();
  }

  // skip other type args, these args don't used in kernel selection
  template <typename T>
  void operator()(const T& x) {
    // do nothing
  }
};

struct KernelTypeParser : ArgsIterator<KernelTypeParser> {
  KernelType kernel_type{KernelType::DENSE_TENSOR_KENREL};

  // TODO(chenweihang): deal with multiple diff input Tensors
  // TODO(chenweihang): add global device guard method to set backend
  void operator()(const Tensor& x) {
    if (phi::SelectedRows::classof(x.impl().get())) {
      kernel_type = KernelType::SELECTED_ROWS_KENREL;
    } else if (phi::SparseCooTensor::classof(x.impl().get())) {
      kernel_type = KernelType::SPARSE_COO_KERNEL;
    } else if (phi::SparseCsrTensor::classof(x.impl().get())) {
      kernel_type = KernelType::SPARSE_CSR_KERNEL;
    }
  }

  // skip other type args, these args don't used in kernel selection
  template <typename T>
  void operator()(const T& x) {
    // do nothing
  }
};

}  // namespace detail

template <typename... Args>
KernelKeySet ParseKernelKeyByInputArgs(const Args&... args) {
  return detail::KernelKeyParser().apply(args...).key_set;
}

template <typename... Args>
KernelType ParseKernelTypeByInputArgs(const Args&... args) {
  return detail::KernelTypeParser().apply(args...).kernel_type;
}

DataType ParseDataType(DataType dtype);
DataType ParseDataType(const Tensor& tensor);
DataType ParseDataType(const std::vector<Tensor>& tensors);
DataType ParseDataTypeWithInputOrder(DataType dtype, const Tensor& tensor);

Backend ParseBackend(const Place& place);
Backend ParseBackend(const Tensor& tensor);
template <typename T, typename... Args>
Backend ParseBackend(T t, Args... args) {
  auto backend_set =
      BackendSet(ParseBackend(t)) | BackendSet(ParseBackend(args...));
  return static_cast<Backend>(64 -
                              detail::CountLeadingZeros(backend_set.bitset()));
}
Backend ParseBackendWithInputOrder(const Place& place, const Tensor& tensor);

DataLayout ParseLayout(DataLayout layout);
DataLayout ParseLayout(const Tensor& tensor);
DataLayout ParseLayoutWithInputOrder(DataLayout layout, const Tensor& tensor);

}  // namespace experimental
}  // namespace paddle
