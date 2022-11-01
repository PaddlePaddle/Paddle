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

struct CustomDeviceProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, CustomDeviceProperties> {
  virtual ~CustomDeviceProperties() = default;
  static const char* name() { return "CustomDeviceProperties"; }

  int64_t storage_format;
  int64_t storage_layout;
};

inline std::unique_ptr<StorageProperties> CopyStorageProperties(
    const std::unique_ptr<StorageProperties>& storage_properties) {
  if (storage_properties) {
    if (CustomDeviceProperties::classof(storage_properties.get())) {
      auto result = std::make_unique<CustomDeviceProperties>();
      result->storage_format =
          static_cast<CustomDeviceProperties*>(storage_properties.get())
              ->storage_format;
      result->storage_layout =
          static_cast<CustomDeviceProperties*>(storage_properties.get())
              ->storage_layout;
      return result;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

}  // namespace phi
