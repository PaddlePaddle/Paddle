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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <optional>

#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"

namespace c10 {
inline Layout layout_or_default(std::optional<Layout> layout) {
  return layout.value_or(kStrided);
}

inline Device device_or_default(std::optional<Device> device) {
  return device.value_or(Device(kCPU));
}
inline ScalarType dtype_or_default(std::optional<ScalarType> dtype) {
  return dtype.value_or(get_default_dtype());
}

inline bool pinned_memory_or_default(std::optional<bool> pinned_memory) {
  return pinned_memory.value_or(false);
}

struct PADDLE_API TensorOptions {
  TensorOptions()
      : requires_grad_(false),
        pinned_memory_(false),
        has_device_(false),
        has_dtype_(false),
        has_layout_(false),
        has_requires_grad_(false),
        has_pinned_memory_(false),
        has_memory_format_(false) {}

  /* implicit */ explicit TensorOptions(Layout layout)  // NOLINT
      : TensorOptions() {
    this->set_layout(layout);
  }

  template <
      typename T,
      typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, Device>>>
  /* implicit */ explicit TensorOptions(T&& device)  // NOLINT
      : TensorOptions() {
    this->set_device(std::forward<T>(device));
  }

  /* implicit */ TensorOptions(c10::ScalarType dtype)  // NOLINT
      : TensorOptions() {
    this->set_dtype(dtype);
  }

  /* implicit */ TensorOptions(MemoryFormat memory_format)  // NOLINT
      : TensorOptions() {
    set_memory_format(memory_format);
  }

  [[nodiscard]] TensorOptions device(
      std::optional<Device> device) const noexcept {
    TensorOptions r = *this;
    r.set_device(device);
    return r;
  }

  [[nodiscard]] TensorOptions device_index(
      c10::DeviceIndex device_index) const noexcept {
    return device(Device(kCUDA, device_index));
  }

  [[nodiscard]] TensorOptions dtype(
      std::optional<ScalarType> dtype) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(dtype);
    return r;
  }

  template <typename T>
  TensorOptions& dtype() {
    has_dtype_ = true;
    return *this;
  }

  [[nodiscard]] TensorOptions layout(
      std::optional<Layout> layout) const noexcept {
    TensorOptions r = *this;
    r.set_layout(layout);
    return r;
  }

  [[nodiscard]] TensorOptions requires_grad(
      std::optional<bool> requires_grad) const noexcept {
    TensorOptions r = *this;
    r.set_requires_grad(requires_grad);
    return r;
  }

  [[nodiscard]] TensorOptions pinned_memory(
      std::optional<bool> pinned_memory) const noexcept {
    TensorOptions r = *this;
    r.set_pinned_memory(pinned_memory);
    return r;
  }

  [[nodiscard]] TensorOptions memory_format(
      std::optional<MemoryFormat> memory_format) const noexcept {
    TensorOptions r = *this;
    r.set_memory_format(memory_format);
    return r;
  }

  Device device() const noexcept { return device_or_default(device_opt()); }

  bool has_device() const noexcept { return has_device_; }

  std::optional<Device> device_opt() const noexcept {
    return has_device_ ? std::make_optional(device_) : std::nullopt;
  }

  c10::DeviceIndex device_index() const noexcept { return device().index(); }

  ScalarType dtype() const noexcept { return dtype_or_default(dtype_opt()); }

  bool has_dtype() const noexcept { return has_dtype_; }

  std::optional<c10::ScalarType> dtype_opt() const noexcept {
    return has_dtype_ ? std::make_optional(dtype_) : std::nullopt;
  }

  Layout layout() const noexcept { return layout_or_default(layout_opt()); }

  bool has_layout() const noexcept { return has_layout_; }

  std::optional<Layout> layout_opt() const noexcept {
    return has_layout_ ? std::make_optional(layout_) : std::nullopt;
  }

  bool requires_grad() const noexcept {
    return has_requires_grad_ ? requires_grad_ : false;
  }

  bool has_requires_grad() const noexcept { return has_requires_grad_; }

  std::optional<bool> requires_grad_opt() const noexcept {
    return has_requires_grad_ ? std::make_optional(requires_grad_)
                              : std::nullopt;
  }

  bool pinned_memory() const noexcept {
    return pinned_memory_or_default(pinned_memory_opt());
  }

  bool has_pinned_memory() const noexcept { return has_pinned_memory_; }

  bool is_sparse() const { return layout_ == c10::Layout::Sparse; }

  bool is_sparse_csr() const { return layout_ == c10::Layout::SparseCsr; }

  bool is_sparse_compressed() const {
    return layout_ == c10::Layout::SparseCsr ||
           layout_ == c10::Layout::SparseCsc ||
           layout_ == c10::Layout::SparseBsr ||
           layout_ == c10::Layout::SparseBsc;
  }

  std::optional<bool> pinned_memory_opt() const noexcept {
    return has_pinned_memory_ ? std::make_optional(pinned_memory_)
                              : std::nullopt;
  }

  bool has_memory_format() const noexcept { return has_memory_format_; }

  std::optional<MemoryFormat> memory_format_opt() const noexcept {
    return has_memory_format_ ? std::make_optional(memory_format_)
                              : std::nullopt;
  }

  TensorOptions merge_memory_format(
      std::optional<MemoryFormat> optional_memory_format) const noexcept {
    TensorOptions merged = *this;
    if (optional_memory_format.has_value()) {
      merged.set_memory_format(optional_memory_format);
    }
    return merged;
  }

  ::phi::Place _PD_GetPlace() const { return device_._PD_GetInner(); }

 private:
  void set_device(std::optional<Device> device) & noexcept {
    if (device) {
      device_ = *device;
      has_device_ = true;
    } else {
      has_device_ = false;
    }
  }

  void set_dtype(std::optional<ScalarType> dtype) & noexcept {
    if (dtype) {
      dtype_ = *dtype;
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  void set_layout(std::optional<Layout> layout) & noexcept {
    if (layout) {
      layout_ = *layout;
      has_layout_ = true;
    } else {
      has_layout_ = false;
    }
  }

  void set_requires_grad(std::optional<bool> requires_grad) & noexcept {
    if (requires_grad) {
      requires_grad_ = *requires_grad;
      has_requires_grad_ = true;
    } else {
      has_requires_grad_ = false;
    }
  }

  void set_pinned_memory(std::optional<bool> pinned_memory) & noexcept {
    if (pinned_memory) {
      pinned_memory_ = *pinned_memory;
      has_pinned_memory_ = true;
    } else {
      has_pinned_memory_ = false;
    }
  }

  void set_memory_format(std::optional<MemoryFormat> memory_format) & noexcept {
    if (memory_format) {
      memory_format_ = *memory_format;
      has_memory_format_ = true;
    } else {
      has_memory_format_ = false;
    }
  }

  Device device_ = c10::kCPU;
  c10::ScalarType dtype_ = c10::ScalarType::Float;
  Layout layout_ = at::kStrided;                           // 8-bit
  MemoryFormat memory_format_ = MemoryFormat::Contiguous;  // 8-bit

  bool requires_grad_ : 1;
  bool pinned_memory_ : 1;

  bool has_device_ : 1;
  bool has_dtype_ : 1;
  bool has_layout_ : 1;
  bool has_requires_grad_ : 1;
  bool has_pinned_memory_ : 1;
  bool has_memory_format_ : 1;
};

inline TensorOptions dtype(ScalarType dtype) {
  return TensorOptions().dtype(dtype);
}

inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);
}

inline TensorOptions device(Device device) {
  return TensorOptions().device(device);
}

inline TensorOptions device_index(c10::DeviceIndex device_index) {
  return TensorOptions().device_index(device_index);
}

inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);
}

inline TensorOptions memory_format(MemoryFormat memory_format) {
  return TensorOptions().memory_format(memory_format);
}

std::ostream& operator<<(std::ostream& stream, const TensorOptions& options);

inline std::string toString(const TensorOptions& options) {
  std::ostringstream stream;
  stream << options;
  return stream.str();
}

}  // namespace c10

namespace at {
using namespace c10;  // NOLINT
}  // namespace at

namespace torch {
using namespace c10;  // NOLINT
}  // namespace torch
