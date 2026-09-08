// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_DNNL
#include "dnnl.hpp"  // NOLINT

#include "paddle/phi/core/storage_properties.h"

namespace phi {

struct OneDNNStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, OneDNNStorageProperties> {
  ~OneDNNStorageProperties() override = default;
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

}  // namespace phi
#endif
