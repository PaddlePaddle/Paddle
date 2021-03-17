/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <glog/logging.h>
#include "acl/acl_rt.h"
#include "paddle/fluid/platform/ascend_npu_info.h"

namespace paddle {
namespace platform {
namespace ascend{

int NPUDevice::GetDeviceCount() {
   uint32_t count = 0;
   aclError status = aclrtGetDeviceCount(&count);
   if(status != 0){
       LOG(ERROR) << "aclrtGetDeviceCount error code:" << status;
       return -1;
   }

   return count;
}


}  // namespace ascend
}  // namespace platform
}  // namespace paddle
