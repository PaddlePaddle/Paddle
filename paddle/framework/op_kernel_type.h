/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/data_layout.h"
#include "paddle/framework/data_type.h"
#include "paddle/framework/library_type.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace framework {

/*

Refer to https://stackoverflow.com/questions/35985960/
c-why-is-boosthash-combine-the-best-way-to-combine-hash-values

*/
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType& key) const {
      int place = key.place_.which();
      int data_type = static_cast<int>(key.data_type_);
      int data_layout = static_cast<int>(key.data_layout_);
      int library_type = static_cast<int>(key.library_type_);

      size_t seed = 0;
      hash_combine(seed, place);
      hash_combine(seed, data_type);
      hash_combine(seed, data_layout);
      hash_combine(seed, library_type);
      return seed;
    }
  };

  proto::DataType data_type_;
  framework::DataLayout data_layout_;
  platform::Place place_;
  framework::LibraryType library_type_;

  OpKernelType(
      proto::DataType data_type, platform::Place place,
      framework::DataLayout data_layout = framework::DataLayout::kAnyLayout,
      framework::LibraryType library_type = framework::LibraryType::kPlain)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(place),
        library_type_(library_type) {}

  OpKernelType(
      proto::DataType data_type, const platform::DeviceContext& dev_ctx,
      framework::DataLayout data_layout = framework::DataLayout::kAnyLayout,
      framework::LibraryType library_type = framework::LibraryType::kPlain)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(dev_ctx.GetPlace()),
        library_type_(library_type) {}

  bool operator==(const OpKernelType& o) const {
    return platform::places_are_same_class(place_, o.place_) &&
           data_type_ == o.data_type_ && data_layout_ == o.data_layout_ &&
           library_type_ == o.library_type_;
  }
};

}  // namespace framework
}  // namespace paddle
