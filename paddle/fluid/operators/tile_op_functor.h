// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <vector>

#include "paddle/fluid/framework/operator.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {

inline std::vector<int> get_repeat_times(
    const framework::ExecutionContext& ctx) {
  if (ctx.HasInput("RepeatTimes")) {
    auto* repeat_tensor = ctx.Input<framework::LoDTensor>("RepeatTimes");
    auto* repeat_data = repeat_tensor->data<int>();
    framework::Tensor cpu_repeat_tensor;
    if (platform::is_gpu_place(repeat_tensor->place()) ||
        platform::is_xpu_place(repeat_tensor->place()) ||
        platform::is_npu_place(repeat_tensor->place())) {
      paddle::framework::TensorCopySync(*repeat_tensor, platform::CPUPlace(),
                                        &cpu_repeat_tensor);
      repeat_data = cpu_repeat_tensor.data<int>();
    }
    auto vec_repeat_times =
        std::vector<int>(repeat_data, repeat_data + repeat_tensor->numel());
    return vec_repeat_times;
  }

  auto list_repeat_times_tensor =
      ctx.MultiInput<framework::Tensor>("repeat_times_tensor");
  if (list_repeat_times_tensor.size() > 0) {
    // get tensor from
    std::vector<int> vec_repeat_times;
    for (size_t i = 0; i < list_repeat_times_tensor.size(); ++i) {
      auto tensor = list_repeat_times_tensor[i];
      if (platform::is_gpu_place(tensor->place()) ||
          platform::is_xpu_place(tensor->place()) ||
          platform::is_npu_place(tensor->place())) {
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_repeat_times.push_back(*temp.data<int32_t>());
      } else {
        vec_repeat_times.push_back(*tensor->data<int32_t>());
      }
    }
    return vec_repeat_times;
  } else {
    return ctx.Attr<std::vector<int>>("repeat_times");
  }
}

}  // namespace operators
}  // namespace paddle
