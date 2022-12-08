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

#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"  // NOLINT
#endif

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

// Add OneDNNStorageProperties firstly for unittest covergae
#ifdef PADDLE_WITH_MKLDNN
struct OneDNNStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, OneDNNStorageProperties> {
  virtual ~OneDNNStorageProperties() = default;
  static const char* name() { return "OneDNNStorageProperties"; }

  /**
   * @brief the detail format of memory block which have layout as ONEDNN
   *
   * @note ONEDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a ONEDNN memory block, layout will be set as
   *       DataLayout::ONEDNN meanwhile detail memory format will be kept in
   *       this field.
   */
  dnnl::memory::format_tag format = dnnl::memory::format_tag::undef;

  /// \brief memory descriptor of tensor which have layout set as ONEDNN
  dnnl::memory::desc mem_desc;
};
#endif

static std::unique_ptr<StorageProperties> CopyStorageProperties(
    const std::unique_ptr<StorageProperties>& sp) {
  if (sp) {
    if (NPUStorageProperties::classof(sp.get())) {
      auto result = std::make_unique<NPUStorageProperties>();
      result->storage_format =
          static_cast<NPUStorageProperties*>(sp.get())->storage_format;
      result->storage_dims =
          static_cast<NPUStorageProperties*>(sp.get())->storage_dims;
      return result;
#ifdef PADDLE_WITH_MKLDNN
    } else if (OneDNNStorageProperties::classof(sp.get())) {
      auto result = std::make_unique<OneDNNStorageProperties>();
      result->format = static_cast<OneDNNStorageProperties*>(sp.get())->format;
      result->mem_desc =
          static_cast<OneDNNStorageProperties*>(sp.get())->mem_desc;
      return result;
#endif
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

}  // namespace phi
