/* copyright (c) 2019 paddlepaddle authors. all rights reserved.

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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class OpInfoMap;
}  // namespace framework
namespace platform {
class DeviceContextPool;
}  // namespace platform
}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

// C-API to get global OpInfo map.
paddle::framework::OpInfoMap &PD_GetOpInfoMap();

// C-API to init global DeviceContextPool from outside.
void PD_InitDevicesPool(paddle::platform::DeviceContextPool *pool);

// C-API to serialize the grad op protocol message to a binary string.
std::vector<std::string> PD_GetGradOpDescStrs(
    const paddle::framework::OpDesc &op_desc,
    const std::unordered_set<std::string> &no_grad_set,
    std::unordered_map<std::string, std::string> *grad_to_var,
    const std::vector<paddle::framework::BlockDesc *> &grad_block);

#ifdef __cplusplus
}
#endif
