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

#include "paddle/pten/core/backend.h"
#include "paddle/pten/core/dtype.h"
#include "paddle/pten/core/layout.h"

// fluid headers [may be replaced by new impl]
#include "paddle/fluid/framework/ddim.h"

namespace pt {

/*
class InplaceVersion {
 public:
 private:
};
*/

/**
 * The Meta data member of TensorImpl.
 * It holds Tensor description information and status information.
 *
 * Note: TensorMeta is a struct, the members are named like
 * ordinary nonmember variables, such as `type` instead of `type_`.
 * And we direct access its members, in addition to constructor, destructor
 * and functions for setting data members, can not provide other functions.
 */
struct TensorMeta {
  TensorMeta() = delete;

  // May introduce bug
  explicit TensorMeta(DDim dims) : dims(dims) {}

  // Compatible Contructor
  TensorMeta(const DDim& dims,
             Backend backend,
             DataType type,
             Layout layout,
             size_t offset)
      : dims(dims),
        backend(backend),
        type(type),
        layout(layout),
        offset(offset) {}

  DDim dims;

  Backend backend{Backend::kCPU};
  DataType type{DataType::kFLOAT32};
  Layout layout{Layout::kNCHW};
  size_t offset{0};

  // InplaceVersion inplace_version_counter{0};
};

}  // namespace pt
