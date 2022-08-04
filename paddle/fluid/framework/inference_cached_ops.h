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
#include <string>
#include <vector>

namespace paddle {
namespace framework {

// cached ops will be captured to accelerate gpu performance.
//      1. op will generate a cudaGraph to record inner gpu kernels
//      2. inner gpu kernels can be launched by calling the cudagraphExecutor
//      only once.
std::vector<std::string> cached_gpu_ops{"conv2d_fusion", "depthwise_conv2d"};

}  // namespace framework
}  // namespace paddle
