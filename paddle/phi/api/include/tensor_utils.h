/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>

#include "paddle/phi/api/include/tensor.h"

namespace paddle {

using Deleter = std::function<void(void*)>;

/**
 * @brief Construct a Tensor from a buffer pointed to by `data`
 *
 * @note `from_blob` doesn’t copy or move data, Modifying the constructed tensor
 *       is equivalent to modifying the original data.
 *
 * @param data The pointer to the memory buffer.
 * @param shape The dims of the tensor.
 * @param dtype The data type of the tensor, should correspond to data type of
 *              `data`. See PD_FOR_EACH_DATA_TYPE in `phi/common/data_type.h`
 * @param layout The data layout of the tensor.
 * @param place The place where the tensor is located.
 *              If `place` is default value, it will be inferred from `data`,
 *              However, the feature is only supported on CPU or GPU.
 *              If `place` is not default value, make sure that `place` is equal
 *              to the place of `data`
 * @param deleter A function or function object that will be called to free the
 *                memory buffer.
 *
 * @return A Tensor object constructed from the buffer
 */
PADDLE_API Tensor from_blob(void* data,
                            const phi::IntArray& shape,
                            phi::DataType dtype,
                            phi::DataLayout layout = phi::DataLayout::NCHW,
                            const phi::Place& place = phi::Place(),
                            const Deleter& deleter = nullptr);

}  // namespace paddle
