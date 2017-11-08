/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

/**
 * @brief   Copy the content of external tensor to a new place.
 *
 * @param[in] src        The external tensor.
 * @param[in] dst_place  The dst place.
 * @param[in] ctx        The device context contains device resources.
 *
 * @note    CopyFrom supports CPU <-> GPU, GPU <-> GPU.
 */
void CopyFrom(const Tensor& src, const platform::Place& dst_place,
              const platform::DeviceContext& ctx, Tensor* dst);

/**
 * @brief   Copy the content of an external vector to a tensor.
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
void CopyFromVector(const std::vector<T>& src,
                    const platform::DeviceContext& ctx, Tensor* dst);

/**
 * @brief   Copy the content of a tensor to a vector
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
void CopyToVector(const Tensor& src, const platform::DeviceContext& ctx,
                  std::vector<T>* dst);

}  // namespace framework
}  // namespace paddle
