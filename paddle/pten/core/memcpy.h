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

#include "paddle/pten/common/place.h"

namespace pten {

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  dst_place Destination allocation place (CPU or GPU or XPU or
 * CustomDevice).
 * \param[in]  dst      Destination memory address.
 * \param[in]  src_place Source allocation place (CPU or GPU or XPU or
 * CustomDevice).
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 * \param[in]  stream   stream for asynchronously memory copy.
 *
 * \note    For GPU/XPU/CustomDevice memory copy, stream need to be specified
 *          for asynchronously memory copy, and type is restored in the
 *          implementation.
 */
void Memcpy(const Place& dst_place,
            void* dst,
            const Place& src_place,
            const void* src,
            size_t num,
            void* stream = nullptr);

}  // namespace pten
