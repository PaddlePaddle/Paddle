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
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"

// TODO(chenweihang): split KernelName, Key, Kernel, Factory into diff files
#include "paddle/pten/core/kernel_factory.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace experimental {

// TODO(shixiaowei): replaced by new DeviceContext later
using CPUContext = paddle::platform::CPUDeviceContext;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
using CUDAContext = paddle::platform::CUDADeviceContext;
#endif

namespace detail {
BackendSet GetTensorBackendSet(const Tensor& t);
std::size_t CountLeadingZeros(uint64_t val);
}  // namespace detail

paddle::platform::DeviceContext* GetDeviceContextByBackend(
    pten::Backend backend);

// TODO(chenweihang): support DataLayout and DataType selected
struct KernelKeySet {
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

  // TODO(chenweihang): deal with multiple diff input Tensors
  // TODO(chenweihang): add global device guard method to set backend
  void operator()(const Tensor& x) {
    key_set.backend_set = key_set.backend_set | detail::GetTensorBackendSet(x);
    // TODO(chenweihang): selecte multi layout and dtype
    key_set.layout = x.layout();
    key_set.dtype = x.type();
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

}  // namespace experimental
}  // namespace paddle
