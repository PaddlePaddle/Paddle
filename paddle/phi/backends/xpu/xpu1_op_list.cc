/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/phi/backends/xpu/xpu_op_list.h"

namespace phi {
namespace backends {
namespace xpu {

XPUOpMap& get_kl1_ops() {
  // KL1支持的op，通过op_name, data_type
  static XPUOpMap s_xpu1_kernels{
      // AddMore
  };

  PD_THROW("get_kl1_ops unsupported");
  return s_xpu1_kernels;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
