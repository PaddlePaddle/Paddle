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
#include <string>

#include "paddle/fluid/platform/xpu/xpu1_op_list.h"
#include "paddle/fluid/platform/xpu/xpu2_op_list.h"
#include "paddle/fluid/platform/xpu/xpu_info.h"
#include "paddle/fluid/platform/xpu/xpu_op_list.h"

namespace paddle {
namespace platform {

bool is_xpu_support_op(std::string op_name, const pOpKernelType& type) {
  auto& ops = get_kl1_ops();
  auto v =
      get_xpu_version(BOOST_GET_CONST(platform::XPUPlace, type.place_).device);
  if (v == XPU2) {
    ops = get_kl2_ops();
  }

  if (ops.find(op_name) != ops.end() &&
      ops[op_name].find(type) != ops[op_name].end()) {
    return true;
  }
  return false;
}

}  // namespace platform
}  // namespace paddle
#endif
