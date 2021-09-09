// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle_infer {
namespace contrib {

class TensorUtils {
 public:
  static void* CudaMallocPinnedMemory(size_t size);
  static void CudaFreePinnedMemory(void* mem);

  static void CopyTensor(Tensor* p_dst, const Tensor& src);
  static void CopyTensorAsync(Tensor* p_dst, const Tensor& src,
                              void* exec_stream);
  static void CopyTensorAsync(Tensor* p_dst, const Tensor& src, CallbackFunc cb,
                              void* cb_params);

 private:
  static void CopyTensorImpl(Tensor* p_dst, const Tensor& src,
                             void* exec_stream, CallbackFunc cb,
                             void* cb_params);
};

class Status {
 public:
  using Code = int;
  Status() noexcept;
  explicit Status(std::exception_ptr e) noexcept;

  Status(const Status&) noexcept;
  Status& operator=(const Status&) noexcept;

  Status& operator=(Status&&) noexcept(
      noexcept(std::declval<std::shared_ptr<Status>>().operator=(
          std::declval<std::shared_ptr<Status>>()))) = default;

  Status(Status&&) noexcept(noexcept(std::shared_ptr<Status>(
      std::declval<std::shared_ptr<Status>>()))) = default;

  static Status OK() noexcept;
  bool ok() const noexcept;
  Code code() const noexcept;
  const std::string& error_message() const noexcept;
  bool operator==(const Status& x) const noexcept;
  bool operator!=(const Status& x) const noexcept;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

template <typename Func, typename... Args>
Status status_wrapper(Func func, Args&&... args) noexcept {
  try {
    func(std::forward<Args>(args)...);
  } catch (...) {
    return Status(std::current_exception());
  }
  return {};
}

}  // namespace contrib
}  // namespace paddle_infer
