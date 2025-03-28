// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <flagcx.h>
#include <string>
#include "paddle/phi/core/distributed/types.h"

namespace phi {
namespace distributed {

#define FLAGCX_CHECK(cmd)                                                   \
  do {                                                                      \
    flagcxResult_t r = cmd;                                                 \
    if (r != flagcxSuccess) {                                               \
      PADDLE_THROW(                                                         \
          common::errors::External("Failed, FlagCX error %s:%d '%s'\n",     \
                                   __FILE__,                                \
                                   __LINE__,                                \
                                   phi::dynload::flagcxGetErrorString(r))); \
    }                                                                       \
  } while (0)

flagcxRedOp_t ToFlagcxRedType(ReduceOp reduction);

std::string SerializeFlagcxUniqueId(const flagcxUniqueId& flagcxID);

std::string FlagcxDTypeToString(flagcxDataType_t dtype);

std::string FlagcxRedTypeToString(flagcxRedOp_t op);

}  // namespace distributed
}  // namespace phi
