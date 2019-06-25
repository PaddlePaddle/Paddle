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
#include "paddle/fluid/lite/opencl/cl_helper.h"

namespace paddle {
namespace lite {

bool InitOpenCLEngine(std::string cl_path);

/// An elementwise_add method to embed OpenCL logic inside, it is used as a
/// black box so that the framework can remain simple.
/// NOTE Currently, these methods are quite expensive, we will optimize them
/// latter.
void elementwise_add(CLHelper* helper, const float* in, const DDim& in_dim,
                     const float* bias, const DDim& bias_dim, float* out,
                     const DDim& out_dim);

void pool(CLHelper* helper, const std::string pooling_type, const int pad_h,
          const int pad_w, const int stride_h, const int stride_w,
          const int ksize_h, const int ksize_w, const float* in,
          const DDim& in_dim, float* out, const DDim& out_dim);

}  // namespace lite
}  // namespace paddle
