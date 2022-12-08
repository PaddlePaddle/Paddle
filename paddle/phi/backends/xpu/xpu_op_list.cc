/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_op_list.h"
#include <glog/logging.h>
#include <mutex>
#include <string>
#include <unordered_set>
#include "paddle/phi/backends/xpu/xpu_info.h"

namespace phi {
namespace backends {
namespace xpu {

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

bool is_in_xpu_black_list(const std::string& fluid_op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_BLACK_LIST"));
        tokenize(ops, ',', &xpu_black_list);
      }
      inited = true;
      VLOG(3) << "XPU Black List: ";
      for (auto iter = xpu_black_list.begin(); iter != xpu_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_black_list.find(fluid_op_name) != xpu_black_list.end()) {
    return true;
  }
  return false;
}

bool is_xpu_support_op(const std::string& fluid_op_name,
                       const phi::DataType type) {
  if (is_in_xpu_black_list(fluid_op_name)) return false;
  auto v = get_xpu_version(0);
  auto& ops = (v == phi::backends::xpu::XPUVersion::XPU1) ? get_kl1_ops()
                                                          : get_kl2_ops();
  if (ops.find(fluid_op_name) != ops.end() &&
      ops[fluid_op_name].find(type) != ops[fluid_op_name].end()) {
    return true;
  }
  return false;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
