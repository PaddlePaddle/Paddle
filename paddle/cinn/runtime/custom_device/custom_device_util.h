// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace custom_device {

/**
 * @brief 通用的自定义设备 Kernel 调用接口。
 * * 该函数不再直接调用特定厂商的 API (如 hipLaunchKernel)，
 * 而是通过 CinnCustomDevicePlugin 转发给厂商插件实现。
 */
void cinn_call_custom_device_kernel(void *kernel_fn,
                                    void *v_args,
                                    int num_args,
                                    int grid_x,
                                    int grid_y,
                                    int grid_z,
                                    int block_x,
                                    int block_y,
                                    int block_z,
                                    int shared_memory_bytes,
                                    void *stream);

/**
 * @brief 用于动态形状推理的 Host 端辅助函数。
 */
void infer_shape_set_value(int row, int col, int64_t value, int64_t **v);

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
