/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
struct EnforceNotMet;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

// insert python call stack & append error op for exception message
void InsertCallStackInfo(const std::string &type, const AttributeMap &attrs,
                         platform::EnforceNotMet *exception);

// only append error op for exception message
void AppendErrorOpHint(const std::string &type,
                       platform::EnforceNotMet *exception);

}  // namespace framework
}  // namespace paddle
