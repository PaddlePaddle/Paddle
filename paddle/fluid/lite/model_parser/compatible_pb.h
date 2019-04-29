// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements the interface to manipute the protobuf message. We use
 * macros to make a compatible interface with the framework::XXDesc and
 * lite::pb::XXDesc.
 */

#include "paddle/fluid/framework/framework.pb.h"
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"
#include "paddle/fluid/lite/model_parser/pb/var_desc.h"
#else
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

namespace paddle {
namespace lite {

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
using OpDesc = lite::pb::OpDesc;
using VarDesc = lite::pb::VarDesc;
#else   // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
using Attribute = framework::Attribute;
using OpDesc = framework::OpDesc;
using VarDesc = framework::VarDesc;
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

}  // namespace lite
}  // namespace paddle
