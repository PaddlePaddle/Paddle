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

#include <string>
#include <utility>

#include "paddle/pten/hapi/include/tensor.h"

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

template <typename Functor>
struct ArgsIterator {
  template <typename... Args>
  inline Functor& apply() {
    return self();
  }

  template <typename T, typename... Args>
  inline Functor& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circurt()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  constexpr bool short_circuit() const { return false; }

 private:
  inline Functor& self() { return *static_cast<Functor*>(this); }
};

struct KernelNameAndKeyParser : ArgsIterator<KernelNameAndKeyParser> {
  std::string kernel_name;
  pten::Backend backend;
  paddle::experimental::DataLayout layout;
  paddle::experimental::DataType dtype;

  explicit KernelNameAndKeyParser(const std::string& name)
      : kernel_name(name) {}

  // TODO(chenweihang): use bit set here
  // TODO(chenweihang): deal with multiple diff input Tensors
  void operator()(const Tensor& x) {
    if (x.is_cpu()) {
      backend = pten::Backend::kCPU;
    } else if (x.is_cuda()) {
      backend = pten::Backend::kCUDA;
    } else {
      throw std::runtime_error("Unsupported backend when parser args.");
    }
    layout = x.layout();
    dtype = x.type();
  }

  // skip other type args
  template <typename T>
  void operator()(const T& x) {
    // do nothing
  }
};

}  // namespace detail

// TODO(chenweihang): Determine the Kernel name and key according to the
// function name and the input Tensor parameters. For example, if the input
// x holds SelectedRows, then the Kernel name should be added with the `sr`
// suffix on the basis of the function name, or the input contains HostTensor,
// and the `host` suffix should be added on the basis of the function name.
template <typename... Args>
std::pair<pten::KernelName, pten::KernelKey> ParseKernelNameAndKeyByArgs(
    const std::string& fn_name, const Args&... args) {
  auto parser = detail::KernelNameAndKeyParser(fn_name);
  parser(args...);
  // TODO(chenweihang): polish design here
  pten::KernelName kernel_name(parser.kernel_name);
  pten::KernelKey kernel_key(parser.backend, parser.layout, parser.dtype);
  return std::make_pair(kernel_name, kernel_key);
}

paddle::platform::DeviceContext* GetDeviceContextByBackend(
    pten::Backend backend) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto place = pten::TransToFluidPlace(backend);
  // switch (backend) {
  //   case Backend::kCPU:
  //     return pool.GetByPlace(paddle::platform::CPUPlace());
  //   case Backend::kCUDA:
  //     return pool.GetByPlace(paddle::platform::CUDAPlace());
  //   default:
  //     throw std::runtime_error(
  //       "Unsupported backend when getting device context.");
  // }
  return pool.Get(place);
}

}  // namespace experimental
}  // namespace paddle
