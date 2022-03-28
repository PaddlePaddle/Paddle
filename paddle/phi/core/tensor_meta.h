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

#include <vector>

#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/utils/any.h"
#include "paddle/utils/optional.h"

namespace phi {

using DDim = phi::DDim;
using LoD = std::vector<std::vector<size_t>>;
/// \brief The meta data of dense tensor. Take the structure type
/// and use all default operations.
///
struct DenseTensorMeta {
  using DataType = paddle::experimental::DataType;
  using DataLayout = paddle::experimental::DataLayout;

  DenseTensorMeta() = default;
  DenseTensorMeta(DataType dtype, const DDim& dims);
  DenseTensorMeta(DataType dtype,
                  const DDim& dims,
                  DataLayout layout,
                  size_t offset = 0);
  DenseTensorMeta(DataType dtype,
                  const DDim& dims,
                  DataLayout layout,
                  const LoD& lod,
                  size_t offset = 0);

  /// \brief Test whether the metadata is valid. Does not throw exceptions.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept;

  bool is_scalar{false};
  DDim dims;
  DataType dtype{DataType::UNDEFINED};
  DataLayout layout{DataLayout::NCHW};
  LoD lod;
  size_t offset{0};
};

inline bool operator==(const DenseTensorMeta& lhs, const DenseTensorMeta& rhs) {
  return (lhs.is_scalar == rhs.is_scalar) && (lhs.dims == rhs.dims) &&
         (lhs.dtype == rhs.dtype) && (lhs.layout == rhs.layout) &&
         (lhs.lod == rhs.lod) && (lhs.offset == rhs.offset);
}

}  // namespace phi
