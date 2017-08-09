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

#include <memory>

namespace paddle {
namespace framework {
namespace details {

/*
 * Slice levels from LOD.
 *
 * @lod: LOD to slice.
 * @level_begin: level to begin slice.
 * @level_end: level to end slice.
 */
std::shared_ptr<LODTensor::LOD> SliceLOD(const LODTensor::LOD &lod,
                                         size_t level_begin, size_t level_end);

/*
 * Slice elements from a level of LOD.
 *
 * @lod: LOD to slice.
 * @level: which level to slice.
 * @elem_begin: element's index to begin slice.
 * @elem_end: element's index to end slice.
 */
std::shared_ptr<LODTensor::LOD> SliceLOD(const LODTensor::LOD &lod,
                                         size_t level, size_t elem_begin,
                                         size_t elem_end, bool tensor_shared);
}  // namespace details
}  // namespace framework
}  // namespace paddle
