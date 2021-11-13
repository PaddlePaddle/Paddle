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

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/ddim.h"
// Note: mixed_vector include many header now, LoD will be
// used on CUDA device? Can we use small_vector here?
// #include "paddle/fluid/framework/mixed_vector.h"

namespace pten {

using DDim = paddle::framework::DDim;
using LoD = std::vector<std::vector<size_t>>;

struct DenseTensorShape {
  using DataType = paddle::experimental::DataType;
  using DataLayout = paddle::experimental::DataLayout;

  DenseTensorShape() = default;
  explicit DenseTensorShape(const DDim& dims);
  DenseTensorShape(const DDim& dims, DataLayout layout);
  DenseTensorShape(const DDim& dims,
                   DataLayout layout,
                   const std::vector<std::vector<size_t>>& lod);

  bool valid() const noexcept;

  DDim dims{std::initializer_list<int64_t>{}};
  DataLayout layout{DataLayout::NCHW};
  LoD lod;
};

inline DenseTensorShape::DenseTensorShape(const DDim& dims) : dims(dims) {}

inline DenseTensorShape::DenseTensorShape(const DDim& dims, DataLayout layout)
    : dims(dims), layout(layout) {}

inline DenseTensorShape::DenseTensorShape(
    const DDim& dims,
    DataLayout layout,
    const std::vector<std::vector<size_t>>& lod)
    : dims(dims), layout(layout), lod(lod) {}

inline bool DenseTensorShape::valid() const noexcept {
  bool valid{true};
  valid = valid && (dims.size() > 0);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (product(dims) >= 0);
  return valid;
}

inline bool operator==(const DenseTensorShape& lhs,
                       const DenseTensorShape& rhs) {
  bool ret = true;
  return ret && (lhs.dims == rhs.dims) && (lhs.layout == rhs.layout) &&
         (lhs.lod == rhs.lod);
}

/// \brief The meta data of dense tensor. Take the structure type
/// and use all default operations.
///
struct DenseTensorMeta {
  DenseTensorMeta() = default;
  DenseTensorMeta(DataType type, const DenseTensorShape& shape);
  DenseTensorMeta(DataType type, DenseTensorShape&& shape);

  /// \brief Test whether the metadata is valid. Does not throw exceptions.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept;

  /// During the entire life cycle of a DenseTensor, the following attributes
  /// marked with `const` are expected to remain unchanged.
  DataType type{DataType::UNDEFINED};
  DenseTensorShape shape;
};

inline DenseTensorMeta::DenseTensorMeta(DataType type,
                                        const DenseTensorShape& shape)
    : type(type), shape(shape) {}

inline DenseTensorMeta::DenseTensorMeta(DataType type, DenseTensorShape&& shape)
    : type(type), shape(std::move(shape)) {}

inline bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (type != DataType::UNDEFINED);
  valid = valid && (shape.valid());
  return valid;
}

inline bool operator==(const DenseTensorMeta& lhs, const DenseTensorMeta& rhs) {
  bool ret = true;
  return ret && (lhs.type == rhs.type) && (lhs.shape == rhs.shape);
}

}  // namespace pten
