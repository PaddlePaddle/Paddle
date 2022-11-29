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

#include "paddle/phi/core/compat/convert_utils.h"

DECLARE_bool(enable_xpu_fast_mode);

namespace phi {
namespace backends {
namespace xpu {

// declaration
XPUOpMap& GetKL2Ops();

// ops_string contains op_list(e.g., 'mul,mul_grad'), parse the op string and
// insert op to op set
static void Tokenize(const std::string& ops,
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

bool IsInXPUBlackList(const std::string& kernel_name) {
  static bool inited = false;
  auto& op_name = TransToFluidOpName(kernel_name);
  VLOG(6) << "fluid_op_name: " << op_name;
  static std::unordered_set<std::string> xpu_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_BLACK_LIST"));
        Tokenize(ops, ',', &xpu_black_list);
      }
      inited = true;
      VLOG(3) << "XPU Black List: ";
      for (auto iter = xpu_black_list.begin(); iter != xpu_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_black_list.find(op_name) != xpu_black_list.end()) {
    return true;
  }
#endif
  return false;
}

bool IsXPUSupportKernel(const std::string& kernel_name) {
  auto& op_name = TransToFluidOpName(kernel_name);
  auto& ops = GetKL2Ops();
  if (ops.find(op_name) != ops.end() &&
      ops[op_name].find(type) != ops[op_name].end()) {
    return true;
  }
  return false;
}

bool IsXPUFallbackToCPU(const std::string& kernel_name, bool kernel_not_exist) {
  if (!FLAGS_enable_xpu_fast_mode) {
    return !IsXPUSupportKernel(kernel_name) || IsInXPUBlackList(kernel_name);
  }
  return kernel_not_exist;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
