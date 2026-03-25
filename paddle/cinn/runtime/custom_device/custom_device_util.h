// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
 * @brief A generic interface for invoking custom device kernels.
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
 * @brief A host-side helper function used for dynamic shape inference.
 */
void infer_shape_set_value(int row, int col, int64_t value, int64_t **v);

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
