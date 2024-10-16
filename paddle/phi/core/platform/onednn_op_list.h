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

#pragma once

#ifdef PADDLE_WITH_DNNL

#include <unordered_set>

namespace paddle {
namespace platform {

// NOTE(jiahongyu): Below ops have specific PADDLE_WITH_DNNL hard codes within
// the function GetExpectedKernelType, so we need to handle them through
// onednn_white_list and solve them one-by-one in the future.
// TODO(jiahongyu): Delete onednn_white_list and fully support
// PADDLE_WITH_DNNL of GetExpectedKernelType.
static const std::unordered_set<std::string> onednn_white_list = {
    // NOTE(jiahongyu): Below ops use mem_desc function, which is encoded by
    // PADDLE_WITH_DNNL in DenseTensor. The hardcodes within
    // GetExpectedKernelType of these ops cannot be deleted now.
    "pad2d",
    "pad3d",
    "slice",
    "slice_grad",
    "split"};

inline bool in_onednn_white_list(const std::string& op_name) {
  return onednn_white_list.find(op_name) != onednn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
