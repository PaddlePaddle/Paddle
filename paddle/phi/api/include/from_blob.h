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
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace experimental {

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
 *              `data`. See PD_FOR_EACH_DATA_TYPE in phi/common/data_type.h
 * @param place The place where the tensor is located, should correspond to
 *              place of `data`.
 * @param layout The data layout of the tensor.
 * @param storage_offset The offset (in bytes) of the tensor data from the
 *                       beginning of `data`.
 * @param deleter A function or function object that will be called to free the
 *                memory buffer.
 *
 * @return A Tensor object constructed from the buffer
 */
PADDLE_API Tensor from_blob(void* data,
                            const phi::DDim& shape,
                            DataType dtype,
                            const Place& place,
                            DataLayout layout,
                            size_t storage_offset,
                            const Deleter& deleter);

inline Tensor from_blob(void* data,
                        const phi::DDim& shape,
                        DataType dtype,
                        const Place& place) {
  return from_blob(data, shape, dtype, place, DataLayout::NCHW, 0, nullptr);
}

}  // namespace experimental
}  // namespace paddle
