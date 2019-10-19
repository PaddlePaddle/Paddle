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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/c/c_api.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/init.h"

extern "C" {

paddle::framework::OpInfoMap &PD_GetOpInfoMap() {
  return paddle::framework::OpInfoMap::Instance();
}

void PD_InitDevicesPool(paddle::platform::DeviceContextPool *pool) {
  paddle::platform::DeviceContextPool::SetPool(pool);
}

std::vector<std::string> PD_GetGradOpDescStrs(
    const paddle::framework::OpDesc &op_desc,
    const std::unordered_set<std::string> &no_grad_set,
    std::unordered_map<std::string, std::string> *grad_to_var,
    const std::vector<paddle::framework::BlockDesc *> &grad_block) {
  auto &op_info = PD_GetOpInfoMap().Get(op_desc.Type());
  std::vector<std::string> ret;
  if (op_info.grad_op_maker_) {
    auto grad_op_descs =
        op_info.grad_op_maker_(op_desc, no_grad_set, grad_to_var, grad_block);
    size_t op_num = grad_op_descs.size();
    ret.resize(op_num);
    for (size_t i = 0; i < op_num; ++i) {
      PADDLE_ENFORCE_EQ(
          grad_op_descs[i]->Proto()->SerializePartialToString(&ret[i]), true,
          "Cannot serialize message.");
    }
  }
  return ret;
}

}  // end extern "C"
