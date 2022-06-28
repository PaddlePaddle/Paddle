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
  static void CopyTensorAsync(Tensor* p_dst,
                              const Tensor& src,
                              void* exec_stream);
  static void CopyTensorAsync(Tensor* p_dst,
                              const Tensor& src,
                              CallbackFunc cb,
                              void* cb_params);

 private:
  static void CopyTensorImpl(Tensor* p_dst,
                             const Tensor& src,
                             void* exec_stream,
                             CallbackFunc cb,
                             void* cb_params);
};

/// \brief A status class, used to intercept exceptions and convert
/// them into a status number.
class Status {
 public:
  using Code = int;
  struct Impl;

  Status();
  explicit Status(std::exception_ptr e);

  Status(const Status&);
  Status& operator=(const Status&) noexcept;
  Status& operator=(Status&&) = default;
  Status(Status&&) = default;

  ///
  /// \brief Construct a status which indicate ok.
  ///
  /// \return A status which indicate ok.
  ///
  static Status OK();

  ///
  /// \brief Determine whether the status is ok.
  ///
  /// \return Whether the status is ok.
  ///
  bool ok() const noexcept;

  ///
  /// \brief Return the error code.
  /// The meaning corresponds to the following.
  ///
  /// CODE    IMPLICATION
  ///  -1      UNKNOWN
  ///  0        NORMAL
  ///  1        LEGACY
  ///  2    INVALID_ARGUMENT
  ///  3       NOT_FOUND
  ///  4     OUT_OF_RANGE
  ///  5    ALREADY_EXISTS
  ///  6   RESOURCE_EXHAUSTED
  ///  7  PRECONDITION_NOT_MET
  ///  8   PERMISSION_DENIED
  ///  9   EXECUTION_TIMEOUT
  ///  10    UNIMPLEMENTED
  ///  11     UNAVAILABLE
  ///  12        FATAL
  ///  13       EXTERNAL
  ///
  /// \return The error code.
  ///
  Code code() const noexcept;

  ///
  /// \brief Return the error message.
  ///
  /// \return The error message.
  ///
  const std::string& error_message() const noexcept;

  bool operator==(const Status& x) const noexcept;
  bool operator!=(const Status& x) const noexcept;

 private:
  std::shared_ptr<Impl> impl_;
};

///
/// \brief A wrapper used to provide exception safety.
///
/// \param func Wrapped function.
/// \param args Parameters of the wrapped function.
/// \return State result of calling function.
///
template <typename Func, typename... Args>
Status get_status(Func func, Args&&... args) noexcept(
    noexcept(Status(std::declval<Status>()))) {
  try {
    func(std::forward<Args>(args)...);
  } catch (...) {
    return Status(std::current_exception());
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace paddle_infer
