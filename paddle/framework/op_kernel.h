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

#include "paddle/"

namespace paddle {
namespace framework {

// define some kernel hint
const std::string kForceCPU = "force_cpu";
const std::string kUseCUDNN = "use_cudnn";
const std::string kUseMKLDNN = "use_mkldnn";

struct OpKernelType {
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const OpKernelType& key) const {
      int place = key.place_.which();
      int data_type = static_cast<int>(key.data_type_);
      int pre_hash = data_type << NUM_PLACE_TYPE_LIMIT_IN_BIT |
                     (place & ((1 << NUM_PLACE_TYPE_LIMIT_IN_BIT) - 1));
      return hash_(pre_hash);
    }
  };

  platform::Place place_;
  proto::DataType data_type_;

  OpKernelType(proto::DataType data_type, platform::Place place)
      : place_(place), data_type_(data_type) {}

  OpKernelType(proto::DataType data_type,
               const platform::DeviceContext& dev_ctx)
      : place_(dev_ctx.GetPlace()), data_type_(data_type) {}

  bool operator==(const OpKernelType& o) const {
    return platform::places_are_same_class(place_, o.place_) &&
           data_type_ == o.data_type_;
  }
};
}
}
