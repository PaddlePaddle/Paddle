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
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace framework {

using DataLayout = phi::DataLayout;

class OpKernelType {
 public:
  constexpr static int kDefaultCustomizedTypeValue = 0;

  // In total should be smaller than 64.
  constexpr static int kPlaceBits = 4;
  constexpr static int kPrimaryDTypeBits = 8;
  constexpr static int kLayoutBits = 4;
  constexpr static int kLibBits = 4;
  constexpr static int kCustomizeBits = 4;

  OpKernelType(proto::VarType::Type data_type,
               platform::Place place,
               DataLayout data_layout = DataLayout::kAnyLayout,
               LibraryType library_type = LibraryType::kPlain,
               int customized_type_value = kDefaultCustomizedTypeValue)
      : data_type_(data_type),
        data_layout_(data_layout),
        place_(place),
        library_type_(library_type),
        customized_type_value_(customized_type_value) {}

  OpKernelType(proto::VarType::Type data_type,
               const phi::DeviceContext& dev_ctx,
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

  bool operator<(const OpKernelType& o) const {
    return hash_key() < o.hash_key();
  }

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
  os << "{data_type[" << kernel_key.data_type_ << "]; data_layout["
     << kernel_key.data_layout_ << "]; place[" << kernel_key.place_
     << "]; library_type[" << kernel_key.library_type_ << "]}";
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
  ret |= (l != DataLayout::ONEDNN && r == DataLayout::ONEDNN);
  ret |= (l == DataLayout::ONEDNN && r != DataLayout::ONEDNN);
#endif
  return ret;
}

inline bool NeedTransformDataType(const phi::KernelKey& l,
                                  const phi::KernelKey& r) {
  return (l.dtype() != r.dtype());
}

inline bool backends_are_same_class(const phi::Backend& l,
                                    const phi::Backend& r) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  size_t num_backends = static_cast<size_t>(phi::Backend::NUM_BACKENDS);
  if (static_cast<size_t>(l) > num_backends &&
      static_cast<size_t>(r) > num_backends) {
    return phi::TransToPhiPlace(l).GetDeviceType() ==
           phi::TransToPhiPlace(r).GetDeviceType();
  }
#endif
  return l == r;
}

inline bool NeedTransform(const phi::KernelKey& l, const phi::KernelKey& r) {
  return (!backends_are_same_class(l.backend(), r.backend())) ||
         (l.dtype() != r.dtype()) ||
         NeedTransformLayout(l.layout(), r.layout());
}

}  // namespace framework
}  // namespace paddle
