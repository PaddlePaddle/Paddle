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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/api/lib/backend_set.h"
#include "paddle/pten/api/lib/data_type_set.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/selected_rows.h"

// TODO(chenweihang): split Key, Kernel, Factory into diff files
#include "paddle/pten/core/kernel_factory.h"

namespace paddle {
namespace experimental {

namespace detail {
BackendSet GetTensorBackendSet(const Tensor& t);
std::size_t CountLeadingZeros(uint64_t val);
}  // namespace detail

pten::DeviceContext* GetDeviceContextByBackend(pten::Backend backend);

enum class KernelType {
  DENSE_TENSOR_KENREL,  // kernel for DenseTensor
  SELECTED_ROWS_KENREL  // kernel for SelectedRows
};

// TODO(chenweihang): support DataLayout and DataType selected
struct KernelKeySet {
  KernelType kernel_type{KernelType::DENSE_TENSOR_KENREL};

  BackendSet backend_set{Backend::UNDEFINED};
  DataLayout layout{DataLayout::UNDEFINED};
  DataType dtype{DataType::UNDEFINED};

  // TODO(chenweihang): iterate all kernelkey for kernel selection
  pten::KernelKey GetHigestPriorityKernelKey() {
    return pten::KernelKey(static_cast<Backend>(64 - detail::CountLeadingZeros(
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
    key_set.backend_set = key_set.backend_set | detail::GetTensorBackendSet(x);
    // TODO(chenweihang): selecte multi layout and dtype
    if (pten::SelectedRows::classof(x.impl().get())) {
      key_set.kernel_type = KernelType::SELECTED_ROWS_KENREL;
    }
    key_set.layout = x.layout();
    key_set.dtype = x.type();
    dtype_set = dtype_set | DataTypeSet(x.dtype());
    auto promote_result = PromoteTypes(dtype_set);
    if (promote_result != DataType::UNDEFINED) {
      key_set.dtype = promote_result;
    }
  }

  void operator()(const std::vector<Tensor>& x) {
    key_set.backend_set =
        key_set.backend_set | detail::GetTensorBackendSet(x[0]);
    // TODO(chenweihang): selecte multi layout and dtype
    key_set.layout = x[0].layout();
    key_set.dtype = x[0].type();
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

DataType ParseDataType(DataType dtype);
DataType ParseDataType(const Tensor& tensor);
DataType ParseDataType(const std::vector<Tensor>& tensors);
DataType ParseDataTypeWithInputOrder(DataType dtype, const Tensor& tensor);

Backend ParseBackend(Backend backend);
Backend ParseBackend(const Tensor& tensor);
template <typename T, typename... Args>
Backend ParseBackend(T t, Args... args) {
  auto backend_set =
      BackendSet(ParseBackend(t)) | BackendSet(ParseBackend(args...));
  return static_cast<Backend>(64 -
                              detail::CountLeadingZeros(backend_set.bitset()));
}
Backend ParseBackendWithInputOrder(Backend backend, const Tensor& tensor);

DataLayout ParseLayout(DataLayout layout);
DataLayout ParseLayout(const Tensor& tensor);
DataLayout ParseLayoutWithInputOrder(DataLayout layout, const Tensor& tensor);

}  // namespace experimental
}  // namespace paddle
