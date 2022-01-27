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
#ifdef PADDLE_WITH_XPU
#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/platform/device/xpu/xpu1_op_list.h"
#include "paddle/fluid/platform/device/xpu/xpu2_op_list.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_kpfirst_list.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"

namespace paddle {
namespace platform {

bool is_xpu_support_op(const std::string& op_name, const pOpKernelType& type) {
  auto& ops = get_kl1_ops();
  auto v = get_xpu_version(type.place_.device);
  if (v == pten::backends::xpu::XPUVersion::XPU2) {
    ops = get_kl2_ops();
  }

  if (ops.find(op_name) != ops.end() &&
      ops[op_name].find(type) != ops[op_name].end()) {
    return true;
  }
  return false;
}

// ops_string contains op_list(e.g., 'mul,mul_grad'), parse the op string and
// insert op to op set
static void tokenize(const std::string& ops, char delim,
                     std::unordered_set<std::string>* op_set) {
  std::string::size_type beg = 0;
  for (uint64_t end = 0; (end = ops.find(delim, end)) != std::string::npos;
       ++end) {
    op_set->insert(ops.substr(beg, end - beg));
    beg = end + 1;
  }

  op_set->insert(ops.substr(beg));
}

bool is_in_xpu_black_list(const std::string& op_name) {
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
  if (xpu_black_list.find(op_name) != xpu_black_list.end()) {
    return true;
  }
  return false;
}

#ifdef PADDLE_WITH_XPU_KP
bool is_xpu_kp_support_op(const std::string& op_name,
                          const pOpKernelType& type) {
  auto& ops = get_kl1_ops();
  auto v = get_xpu_version(type.place_.device);
  if (v == pten::backends::xpu::XPUVersion::XPU2) {
    ops = get_kp_ops();
  }

  if (ops.find(op_name) != ops.end() &&
      ops[op_name].find(type) != ops[op_name].end()) {
    return true;
  }
  return false;
}

bool is_in_xpu_kpwhite_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_kpwhite_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_KPWHITE_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_KPWHITE_LIST"));
        tokenize(ops, ',', &xpu_kpwhite_list);
      }
      inited = true;
      VLOG(3) << "XPU kpwhite List: ";
      for (auto iter = xpu_kpwhite_list.begin(); iter != xpu_kpwhite_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_kpwhite_list.find(op_name) != xpu_kpwhite_list.end()) {
    return true;
  }
  return false;
}
#endif

std::vector<vartype::Type> get_xpu_op_support_type(
    const std::string& op_name, pten::backends::xpu::XPUVersion version) {
  std::vector<vartype::Type> res;
  auto& ops = version == pten::backends::xpu::XPUVersion::XPU1 ? get_kl1_ops()
                                                               : get_kl2_ops();
  if (ops.find(op_name) != ops.end()) {
    XPUKernelSet& type_set = ops[op_name];
    for (auto& item : type_set) {
      res.push_back(item.data_type_);
    }
  }
  return res;
}

XPUOpListMap get_xpu_op_list(pten::backends::xpu::XPUVersion version) {
  XPUOpListMap res;
  auto& ops = version == pten::backends::xpu::XPUVersion::XPU1 ? get_kl1_ops()
                                                               : get_kl2_ops();
  for (auto& op : ops) {
    std::vector<vartype::Type> op_vartypes;
    for (auto& item : op.second) {
      op_vartypes.push_back(item.data_type_);
    }
    res[op.first] = std::move(op_vartypes);
  }
  return res;
}
}  // namespace platform
}  // namespace paddle
#endif
