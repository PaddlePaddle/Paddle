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
#include <vector>
#include "paddle/fluid/framework/experimental/include/common.h"

namespace paddle {
namespace experimental {

template <typename T>
std::shared_ptr<void> make_tensor_impl(std::unique_ptr<T>&& impl);

/// \brief The Tensor interface that facilitates imperative execution
/// scheduling and encapsulation.
/// Different from the default construction behavior, it is implemented
/// here as a shallow copy with a reference count.
///
class Tensor final {
 public:
  using Shape = std::vector<int64_t>;

  Tensor() = default;

  int64_t numel() const;
  const Shape shape() const;
  DataType data_type() const;
  DataLayout layout() const;

  template <typename T>
  T* data() const;

  template <typename T>
  T* mutable_data();

  bool valid() const;
  bool initialized() const;

  bool is_dense_tensor() const;
  bool is_selected_rows() const;

  bool is_cpu() const;
  bool is_cuda() const;

  Tensor cpu() const;
  Tensor cuda() const;

 public:
  // These members should not be used by end users and are implementation
  // details.
  struct Utils;
  template <typename T>
  Tensor(std::unique_ptr<T>&& impl)
      : impl_(make_tensor_impl(std::move(impl))) {}

 private:
  // STL promises that as long as the constructor meets the requirements,
  // resources will not be leaked during destructuring. So it is assumed
  // that the shared pointer supports void type erasure.
  std::shared_ptr<void> impl_;
};

}  // namespace experimental
}  // namespace paddle
