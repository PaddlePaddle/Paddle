// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/backends/cuda_util.h"

#include <glog/logging.h>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"

namespace cinn {
namespace backends {

std::string cuda_thread_axis_name(int level) {
  switch (level) {
    case 0:
      return "threadIdx.x";
      break;
    case 1:
      return "threadIdx.y";
      break;
    case 2:
      return "threadIdx.z";
      break;
  }
  return "";
}

std::string cuda_block_axis_name(int level) {
  switch (level) {
    case 0:
      return "blockIdx.x";
      break;
    case 1:
      return "blockIdx.y";
      break;
    case 2:
      return "blockIdx.z";
      break;
  }
  return "";
}

}  // namespace backends
}  // namespace cinn
