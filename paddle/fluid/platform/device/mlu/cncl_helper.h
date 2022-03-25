/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_CNCL
#include <cncl.h>

#include <stdio.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {

inline cnclDataType_t ToCNCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return cnclFloat32;
  } else if (type == framework::proto::VarType::FP16) {
    return cnclFloat16;
  } else if (type == framework::proto::VarType::INT32) {
    return cnclInt32;
  } else if (type == framework::proto::VarType::INT16) {
    return cnclInt16;
  } else if (type == framework::proto::VarType::INT8) {
    return cnclInt8;
  } else if (type == framework::proto::VarType::UINT8) {
    return cnclUint8;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in cncl is not supported."));
  }
}

}  // namespace platform
}  // namespace paddle
#endif
