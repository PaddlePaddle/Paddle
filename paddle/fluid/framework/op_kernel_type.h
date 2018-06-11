/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/library_type.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType& key) const {
      int place = key.place_.which();
      int data_type = static_cast<int>(key.data_type_) << LEFT_SHIFT;
      int data_layout = static_cast<int>(key.data_layout_) << (LEFT_SHIFT * 2);
      int library_type = static_cast<int>(key.library_type_)
                         << (LEFT_SHIFT * 3);

      std::hash<int> hasher;
      return hasher(place + data_type + data_layout + library_type);
    }
  };

  // place, data_type, library_type kinds less than 2^8
  constexpr static int LEFT_SHIFT = 8;

  proto::VarType::Type data_type_;
  DataLayout data_layout_;
  platform::Place place_;
  LibraryType library_type_;

  OpKernelType(proto::VarType::Type data_type, platform::Place place,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(place),
        library_type_(library_type) {}

  OpKernelType(proto::VarType::Type data_type,
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

  bool operator!=(const OpKernelType& o) const { return !(*this == o); }
};

inline std::ostream& operator<<(std::ostream& os,
                                const OpKernelType& kernel_key) {
  os << "data_type[" << kernel_key.data_type_ << "]:data_layout["
     << kernel_key.data_layout_ << "]:place[" << kernel_key.place_
     << "]:library_type[" << kernel_key.library_type_ << "]";
  return os;
}

inline std::string KernelTypeToString(const OpKernelType& kernel_key) {
  std::ostringstream stream;
  stream << kernel_key;
  return stream.str();
}

inline bool NeedTransformLayout(const DataLayout& l, const DataLayout& r) {
  bool ret =
      (l != DataLayout::kAnyLayout && r != DataLayout::kAnyLayout && l != r);
#ifdef PADDLE_WITH_MKLDNN
  // Layout transform needed for either non-MKLDNN to MKLDNN or vice versa
  ret |= (l != DataLayout::kMKLDNN && r == DataLayout::kMKLDNN);
  ret |= (l == DataLayout::kMKLDNN && r != DataLayout::kMKLDNN);
#endif
  return ret;
}

inline bool TransFromNeeded(const OpKernelType& l, const OpKernelType& r) {
  return (!platform::places_are_same_class(l.place_, r.place_)) ||
         (l.data_type_ != r.data_type_) ||
         NeedTransformLayout(l.data_layout_, r.data_layout_);
}

}  // namespace framework
}  // namespace paddle
