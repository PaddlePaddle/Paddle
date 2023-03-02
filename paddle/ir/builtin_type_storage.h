// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/type.h"

namespace std {
///
/// \brief Enable hashing std::vector<T> instances.
///
template <typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T> &dim) const {
    std::size_t seed = 0;
    for (size_t i = 0; i < dim.size(); ++i) {
      seed ^= std::hash<T>()(dim[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

///
/// \brief Enable hashing std::vector<std::vector<T> instances.
///
template <typename T>
struct hash<std::vector<std::vector<T>>> {
  std::size_t operator()(const std::vector<std::vector<T>> &lod) const {
    std::size_t seed = 0;
    for (size_t i = 0; i < lod.size(); ++i) {
      seed ^= std::hash<std::vector<T>>()(lod[i]) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

}  // namespace std

namespace ir {
///
/// \brief It is consistent with the DataLayout defined by Phi operator library.
/// See the file for details: paddle/phi/common/layout.h.
///
enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  NCDHW,
  NDHWC,
  ONEDNN,
  SPARSE_COO,
  SPARSE_CSR,
  PSTRING_UNION,

  NUM_DATA_LAYOUTS,

  // See Note [ Why we need ALL in basic kernel key member? ]
  ALL_LAYOUT = UNDEFINED,

  // Note: Unify phi DataLayout and fluid::framework::DataLayout,
  // for compatible with fluid DataLayout, here need prefix `k`

  // Note: The original `kAnyLayout (enum value 2)` is a strange design.
  // `kAnyLayout` originally cannot represent any kind of Layout,
  // at the same time, it can also represent any Layout.
  // Strictly, it means "default" or "undefined" layout,
  // and should not be mixed with other meaningful layouts

  kAnyLayout = ANY,
  kNHWC = NHWC,
  kNCHW = NCHW,
  kMKLDNN = ONEDNN,  // all layouts supported by ONEDNN internally
  kNDHWC = NDHWC,
  kNCDHW = NCDHW,
};

///
/// \brief Define Parameteric TypeStorage for DenseTensorType. Derived
/// TypeStorage need to declare ParamKey, define Construction method, define
/// HashValue method, and overload operator==.
///
struct DenseTensorTypeStorage : public ir::TypeStorage {
  using Dim = std::vector<int64_t>;

  using LoD = std::vector<std::vector<size_t>>;

  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<ir::Type, Dim, DataLayout, LoD, size_t>;

  DenseTensorTypeStorage(
      ir::Type dtype, Dim dims, DataLayout layout, LoD lod, size_t offset)
      : dtype_(dtype),
        dims_(dims),
        layout_(layout),
        lod_(lod),
        offset_(offset) {}

  ///
  /// \brief Each derived TypeStorage must define a Construc method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static DenseTensorTypeStorage *Construct(ParamKey key) {
    return new DenseTensorTypeStorage(std::get<0>(key),
                                      std::get<1>(key),
                                      std::get<2>(key),
                                      std::get<3>(key),
                                      std::get<4>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    // hash dtype
    hash_value =
        hash_combine(hash_value, std::hash<ir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = hash_combine(hash_value, std::hash<Dim>()(std::get<1>(key)));
    // hash layout
    hash_value =
        hash_combine(hash_value, std::hash<DataLayout>()(std::get<2>(key)));
    // hash lod
    hash_value = hash_combine(hash_value, std::hash<LoD>()(std::get<3>(key)));
    // hash offset
    hash_value =
        hash_combine(hash_value, std::hash<size_t>()(std::get<4>(key)));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const {
    return ParamKey(dtype_, dims_, layout_, lod_, offset_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(dtype_, dims_, layout_, lod_, offset_);
  }

  ///
  /// \brief DenseTensorTypeStorage include five parameters: dims, dtype,
  /// layout, lod, offset.
  ///
  ir::Type dtype_;
  Dim dims_;
  DataLayout layout_;
  LoD lod_;
  size_t offset_;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

}  // namespace ir
