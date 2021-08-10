/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/npu_blacklist_op.h"

#include <mutex>
#include <string>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace platform {

bool is_in_npu_black_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> npu_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("NPU_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("NPU_BLACK_LIST"));
        npu_black_list = paddle::string::SplitToSet(ops, ',');
      }
      inited = true;
      VLOG(3) << "NPU Black List: ";
      for (auto iter = npu_black_list.begin(); iter != npu_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (npu_black_list.find(op_name) != npu_black_list.end()) {
    return true;
  }
  return false;
}

}  // namespace platform
}  // namespace paddle
#endif
