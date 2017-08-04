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

#include "paddle/framework/lod_tensor.h"

#include <memory>

namespace paddle {
namespace framework {
namespace details {

using LOD = LODTensor::LOD;

std::shared_ptr<LOD> SliceLOD(const LOD &lod, size_t level_begin,
                              size_t level_end) {
  auto new_lod = std::make_shared<LOD>();
  new_lod->reserve(level_end - level_begin);
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod->emplace_back(lod[i]);
  }
  return new_lod;
}

std::shared_ptr<LOD> SliceLOD(const LOD &lod, size_t level, size_t elem_begin,
                              size_t elem_end, bool tensor_shared) {
  // slice the lod.
  auto new_lod = std::make_shared<LOD>();
  new_lod->reserve(lod.size() - level);
  auto start = lod.at(level)[elem_begin];
  auto end = lod.at(level)[elem_end];

  for (auto it = lod.begin() + level; it != lod.end(); it++) {
    auto it_begin = std::find(it->begin(), it->end(), start);
    auto it_end = std::find(it_begin, it->end(), end);
    PADDLE_ENFORCE(it_begin != it->end(), "error in parsing lod info");
    PADDLE_ENFORCE(it_end != it->end(), "error in parsing lod info");
    new_lod->emplace_back(it_begin, it_end + 1);
    if (!tensor_shared) {
      // reset offset if tensor is copyed and sliced.
      std::transform(new_lod->back().begin(), new_lod->back().end(),
                     new_lod->back().begin(),
                     [start](int v) { return v - start; });
      PADDLE_ENFORCE(new_lod->back().front() == 0, "error in slice LOD");
    }
  }
  return new_lod;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
