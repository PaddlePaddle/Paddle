/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_info.h"

namespace paddle::framework {

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
OpInfoMap& OpInfoMap::Instance() {
  static OpInfoMap g_op_info_map;
  return g_op_info_map;
}

std::vector<std::string> OpInfoMap::GetUseDefaultGradOpDescMakerOps() const {
  // Use set to sort op names
  std::set<std::string> result_ops;
  for (auto& pair : map_) {
    if (pair.second.use_default_grad_op_desc_maker_) {
      result_ops.insert(pair.first);
    }
  }
  return std::vector<std::string>(result_ops.begin(), result_ops.end());
}

}  // namespace paddle::framework
