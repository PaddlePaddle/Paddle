/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/custom/custom_device_op_list.h"
#include <glog/logging.h>
#include <mutex>
#include <string>
#include <unordered_set>
namespace phi {
namespace backends {
namespace custom_device {
// ops_string contains op_list(e.g., 'mul,mul_grad'), parse the op string and
// insert op to op set
static void tokenize(const std::string& ops,
                     char delim,
                     std::unordered_set<std::string>* op_set) {
  std::string::size_type beg = 0;
  for (uint64_t end = 0; (end = ops.find(delim, end)) != std::string::npos;
       ++end) {
    op_set->insert(ops.substr(beg, end - beg));
    beg = end + 1;
  }
  op_set->insert(ops.substr(beg));
}

bool is_in_custom_black_list(const std::string& fluid_op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> cs_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("CUSTOM_DEVICE_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("CUSTOM_DEVICE_BLACK_LIST"));
        tokenize(ops, ',', &cs_black_list);
      }
      inited = true;
      VLOG(3) << "Custom Device Black List: ";
      for (auto iter = cs_black_list.begin(); iter != cs_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (cs_black_list.find(fluid_op_name) != cs_black_list.end()) {
    return true;
  }
  return false;
}
}  // namespace custom_device
}  // namespace backends
}  // namespace phi
#endif
