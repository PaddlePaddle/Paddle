/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_kernel_type.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

size_t OpKernelType::Hash::operator()(const OpKernelType& key) const {
  int cur_loc = 0;

  int place = static_cast<int>(key.place_.GetType());
  cur_loc += OpKernelType::kPlaceBits;

  int data_type = static_cast<int>(key.data_type_) << cur_loc;
  cur_loc += OpKernelType::kPrimaryDTypeBits;

  int data_layout = static_cast<int>(key.data_layout_) << cur_loc;
  cur_loc += OpKernelType::kLayoutBits;

  int library_type = static_cast<int>(key.library_type_) << cur_loc;
  cur_loc += OpKernelType::kLibBits;

  int customized_value = key.customized_type_value_;
  PADDLE_ENFORCE_LT(customized_value,
                    (1 << OpKernelType::kCustomizeBits),
                    common::errors::Unavailable(
                        "Too many custom OpKernel attribute values, expected "
                        "maximum value is %d, received value is %d.",
                        (1 << OpKernelType::kCustomizeBits),
                        customized_value));
  customized_value = customized_value << cur_loc;
  cur_loc += OpKernelType::kCustomizeBits;
  PADDLE_ENFORCE_LT(cur_loc,
                    64,
                    common::errors::Unavailable(
                        "Too many OpKernel attribute values, expected maximum "
                        "value is 64, received value is %d.",
                        cur_loc));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  std::hash<int> hasher;
  size_t seed =
      hasher(place + data_type + data_layout + library_type + customized_value);
  if (phi::is_custom_place(key.place_)) {
    seed ^= std::hash<std::string>{}(key.place_.GetDeviceType()) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2) + 4;
  }
  return seed;
#else
  std::hash<int> hasher;
  return hasher(place + data_type + data_layout + library_type +
                customized_value);
#endif
}

bool OpKernelType::operator==(const OpKernelType& o) const {
  return phi::places_are_same_class(place_, o.place_) &&
         data_type_ == o.data_type_ && data_layout_ == o.data_layout_ &&
         library_type_ == o.library_type_ &&
         customized_type_value_ == o.customized_type_value_;
}

}  // namespace paddle::framework
