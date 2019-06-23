/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/opencl/cl_context.h"

namespace paddle {
namespace lite {

bool InitOpenCLEngine(std::string cl_path);
void elementwise_add(CLContext* context, float* in, const DDim& in_dim,
                     float* bias, const DDim& bias_dim, float* out,
                     const DDim& out_dim);

}  // namespace lite
}  // namespace paddle
