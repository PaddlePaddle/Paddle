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

#include "paddle/fluid/platform/cudnn_workspace_helper.h"

#include <cstdlib>
#include "boost/lexical_cast.hpp"
namespace paddle {
namespace platform {

static int GetDefaultConvWorkspaceSizeLimitMBImpl() {
  const char *env_str = std::getenv("FLAGS_conv_workspace_size_limit");
  return env_str ? boost::lexical_cast<int>(std::string(env_str))
                 : kDefaultConvWorkspaceSizeLimitMB;
}

int GetDefaultConvWorkspaceSizeLimitMB() {
  static auto workspace_size = GetDefaultConvWorkspaceSizeLimitMBImpl();
  return workspace_size;
}

}  // namespace platform
}  // namespace paddle
