// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <ATen/core/Tensor.h>

#include "paddle/phi/api/include/tensor_utils.h"
namespace at {

namespace detail {

inline void noopDelete(void* /*unused*/) {}

}  // namespace detail

class TensorMaker {
  friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;

 public:
  using ContextDeleter = DeleterFnPtr;

  TensorMaker& strides(OptionalIntArrayRef value) noexcept {
    strides_ = value;

    return *this;
  }

  TensorMaker& storage_offset(std::optional<int64_t> value) noexcept {
    storage_offset_ = value;

    return *this;
  }

  TensorMaker& deleter(std::function<void(void*)> value) noexcept {
    deleter_ = std::move(value);

    return *this;
  }

  TensorMaker& context(void* value, ContextDeleter deleter = nullptr) noexcept {
    ctx_ = std::unique_ptr<void, ContextDeleter>{
        value, deleter != nullptr ? deleter : detail::noopDelete};

    return *this;
  }

  TensorMaker& target_device(std::optional<Device> value) noexcept {
    device_ = value;

    return *this;
  }

  TensorMaker& options(TensorOptions value) noexcept {
    opts_ = value;

    return *this;
  }

  TensorMaker& resizeable_storage() noexcept {
    resizeable_ = true;

    return *this;
  }

  Tensor make_tensor() {
    PD_CHECK(!deleter_ || !ctx_,
             "The deleter and context arguments are mutually exclusive.");

    PD_CHECK(!storage_offset_.has_value() || storage_offset_.value() == 0,
             "storage_offset` should be zero.");

    if (device_.has_value() && opts_.has_device() &&
        opts_.device().has_index()) {
      PD_CHECK(opts_.device() == *device_,
               "Specified device ",
               opts_.device(),
               " does not match device of data ",
               *device_);
    }

    phi::Place pd_place;
    if (device_.has_value()) {
      pd_place = device_->_PD_GetInner();
    } else if (opts_.has_device() && opts_.device().has_index()) {
      pd_place = opts_.device()._PD_GetInner();
    } else {
      pd_place = phi::Place();  // UNDEFINED → auto-detect inside from_blob
    }

    // Build paddle deleter: prefer explicit deleter_, then wrap ctx_ so its
    // lifetime is tied to the tensor allocation.
    paddle::Deleter pd_deleter = nullptr;
    if (deleter_) {
      pd_deleter = deleter_;
    } else if (ctx_) {
      // shared_ptr takes ownership of the context and calls its deleter when
      // the last copy (held in the lambda) is destroyed.
      auto shared_ctx =
          std::shared_ptr<void>(ctx_.release(), ctx_.get_deleter());
      pd_deleter = [shared_ctx](void* /*data*/) {};
    }

    if (strides_.has_value()) {
      return paddle::from_blob(
          data_,
          sizes_._PD_ToPaddleIntArray(),
          strides_.value()._PD_ToPaddleIntArray(),
          compat::_PD_AtenScalarTypeToPhiDataType(opts_.dtype()),
          phi::DataLayout::NCHW,
          pd_place,
          pd_deleter);
    } else {
      return paddle::from_blob(
          data_,
          sizes_._PD_ToPaddleIntArray(),
          compat::_PD_AtenScalarTypeToPhiDataType(opts_.dtype()),
          phi::DataLayout::NCHW,
          pd_place,
          pd_deleter);
    }
  }

 private:
  explicit TensorMaker(void* data, IntArrayRef sizes) noexcept
      : data_{data}, sizes_{sizes} {}

  std::size_t computeStorageSize() const noexcept;

  DataPtr makeDataPtrFromDeleter() noexcept;

  DataPtr makeDataPtrFromContext() noexcept;

  IntArrayRef makeTempSizes() const noexcept;

  void* data_;
  IntArrayRef sizes_;
  OptionalIntArrayRef strides_;
  std::optional<int64_t> storage_offset_;
  std::function<void(void*)> deleter_;
  std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
  std::optional<Device> device_;
  TensorOptions opts_;
  bool resizeable_{};
};

inline TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept {
  return TensorMaker{data, sizes};
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return for_blob(data, sizes)
      .strides(strides)
      .deleter(deleter)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t storage_offset,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return for_blob(data, sizes)
      .strides(strides)
      .storage_offset(storage_offset)
      .deleter(deleter)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return for_blob(data, sizes)
      .deleter(std::move(deleter))
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(void* data,
                        IntArrayRef sizes,
                        IntArrayRef strides,
                        const TensorOptions& options = {}) {
  return for_blob(data, sizes).strides(strides).options(options).make_tensor();
}

inline Tensor from_blob(void* data,
                        IntArrayRef sizes,
                        const TensorOptions& options = {}) {
  return for_blob(data, sizes).options(options).make_tensor();
}

}  // namespace at
