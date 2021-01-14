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

class OpKernelType {
 public:
  constexpr static int kDefaultCustomizedTypeValue = 0;

  // In total should be smaller than 64.
  constexpr static int kPlaceBits = 4;
  constexpr static int kPrimaryDTypeBits = 8;
  constexpr static int kLayoutBits = 4;
  constexpr static int kLibBits = 4;
  constexpr static int kCustomizeBits = 4;

  OpKernelType(proto::VarType::Type data_type, platform::Place place,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain,
               int customized_type_value = kDefaultCustomizedTypeValue)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(place),
        library_type_(library_type),
        customized_type_value_(customized_type_value) {}

  OpKernelType(proto::VarType::Type data_type,
               const platform::DeviceContext& dev_ctx,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain,
               int customized_type_value = kDefaultCustomizedTypeValue)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(dev_ctx.GetPlace()),
        library_type_(library_type),
        customized_type_value_(customized_type_value) {}

  virtual ~OpKernelType() {}

  struct Hash {
    size_t operator()(const OpKernelType& key) const;
  };

  size_t hash_key() const { return Hash()(*this); }

  bool operator==(const OpKernelType& o) const;

  bool operator!=(const OpKernelType& o) const { return !(*this == o); }

  proto::VarType::Type data_type_;
  DataLayout data_layout_;
  platform::Place place_;
  LibraryType library_type_;
  int customized_type_value_;
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

inline bool NeedTransformDataType(const OpKernelType& l,
                                  const OpKernelType& r) {
  return (l.data_type_ != r.data_type_);
}

inline bool NeedTransform(const OpKernelType& l, const OpKernelType& r) {
  return (!platform::places_are_same_class(l.place_, r.place_)) ||
         (l.data_type_ != r.data_type_) ||
         NeedTransformLayout(l.data_layout_, r.data_layout_);
}

}  // namespace framework
}  // namespace paddle
