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

#include <memory>

namespace paddle {
namespace operators {

///
/// @brief      Helper class for AXPY execution using oneDNN library.
///
/// @tparam     T     Data type.
///
template <typename T>
class OneDNNAXPYHandler {
 public:
  OneDNNAXPYHandler(OneDNNAXPYHandler&) = delete;
  OneDNNAXPYHandler(OneDNNAXPYHandler&&) = delete;
  OneDNNAXPYHandler& operator=(OneDNNAXPYHandler&) = delete;
  OneDNNAXPYHandler& operator=(OneDNNAXPYHandler&&) = delete;
  ///
  /// @brief      Constructor.
  ///
  /// @param[in]  n      The number of elements in tensor (assumed 1D tensor)
  /// @param[in]  alpha  The alpha coefficient.
  ///
  OneDNNAXPYHandler(int64_t n, T alpha);
  ///
  /// @brief      Executes AXPY.
  ///
  /// @param[in]  x     The pointer to input X tensor data.
  /// @param[out] y     The pointer to output Y tensor data.
  ///
  void operator()(const T* x, T* y);

 private:
  OneDNNAXPYHandler() = delete;
  // (arogowie-intel) Private implementation idiom to hide dependency
  // on OneDNN headers.
  class Impl;
  // We need custom deleter, since the compiler is unable to parameterize
  // an allocator's default deleter due to incomple type.
  std::unique_ptr<Impl, void (*)(Impl*)> pimpl_;
};

}  // namespace operators
}  // namespace paddle
