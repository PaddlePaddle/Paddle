/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/utils/type_registry.h"

namespace phi {

struct StorageProperties {
 public:
  virtual ~StorageProperties() = default;

  TypeInfo<StorageProperties> type_info() const { return type_info_; }

 private:
  template <typename T, typename U>
  friend class TypeInfoTraits;

  TypeInfo<StorageProperties> type_info_{
      TypeInfo<StorageProperties>::kUnknownType};
};

struct NPUStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, NPUStorageProperties> {
  virtual ~NPUStorageProperties() = default;
  static const char* name() { return "NPUStorageProperties"; }

  int64_t storage_format{-1};
  DDim storage_dims;
};

#ifdef PADDLE_WITH_XPU
struct XPUStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, XPUStorageProperties> {
  XPUStorageProperties() = default;
  explicit XPUStorageProperties(float value) : xpu_scale_value(value) {}
  virtual ~XPUStorageProperties() = default;
  static const char* name() { return "XPUStorageProperties"; }
  static constexpr float default_xpu_scale_value = -1.0f;

  float xpu_scale_value{default_xpu_scale_value};
};
#endif

std::unique_ptr<StorageProperties> CopyStorageProperties(
    const std::unique_ptr<StorageProperties>& sp);

}  // namespace phi
