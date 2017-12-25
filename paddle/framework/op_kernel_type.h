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
#include "paddle/platform/device_context.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace framework {

struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType& key) const {
      int place = key.place_.which() + (1 << LEFT_SHIFT);
      int data_type =
          static_cast<int>(key.data_type_) + (1 << (LEFT_SHIFT + 1));
      int data_layout =
          static_cast<int>(key.data_layout_) + (1 << (LEFT_SHIFT + 2));
      int library_type =
          static_cast<int>(key.library_type_) + (1 << (LEFT_SHIFT + 3));
      std::hash<int> hasher;
      return hasher(place + data_type + data_layout + library_type);
    }
  };

  // place, data_type, library_type kinds less than 2^8
  constexpr static int LEFT_SHIFT = 8;
  proto::DataType data_type_;
  DataLayout data_layout_;
  platform::Place place_;
  LibraryType library_type_;

  OpKernelType(proto::DataType data_type, platform::Place place,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(place),
        library_type_(library_type) {}

  OpKernelType(proto::DataType data_type,
               const platform::DeviceContext& dev_ctx,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain)
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

inline std::ostream& operator<<(std::ostream& os,
                                const OpKernelType& kernel_key) {
  os << "data_type[" << kernel_key.data_type_ << "]:data_layout["
     << kernel_key.data_layout_ << "]:place[" << kernel_key.place_
     << "]:library_type[" << kernel_key.library_type_ << "]";
  return os;
}

}  // namespace framework
}  // namespace paddle
