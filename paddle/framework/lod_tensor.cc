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

#include <glog/logging.h>

namespace paddle {
namespace framework {

LODTensor LODTensor::SliceShared(size_t level_begin, size_t level_end) const {
  PADDLE_ENFORCE(has_lod(), "has no LOD info, can't be sliced.");
  auto new_lod = details::SliceLOD(*lod_start_pos_, level_begin, level_end);
  return LODTensor(tensor_, new_lod);
}

namespace details {

std::shared_ptr<LODTensor::lod_t> SliceLOD(const LODTensor::lod_t &lod,
                                           size_t level_begin,
                                           size_t level_end) {
  auto new_lod = std::make_shared<LODTensor::lod_t>();
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod->emplace_back(lod[i]);
  }
  return new_lod;
}

std::shared_ptr<LODTensor::lod_t> SliceLOD(const LODTensor::lod_t &lod,
                                           size_t level, size_t elem_begin,
                                           size_t elem_end) {
  // slice the lod.
  auto new_lod = std::make_shared<LODTensor::lod_t>();
  auto start = lod.at(level)[elem_begin];
  auto end = lod.at(level)[elem_end];

  LOG(INFO) << "start: " << start << " end: " << end;

  for (auto it = lod.begin() + level; it != lod.end(); it++) {
    auto it_begin = std::find(it->begin(), it->end(), start);
    auto it_end = std::find(it_begin, it->end(), end);
    PADDLE_ENFORCE(it_begin != it->end(), "error in parsing lod info");
    PADDLE_ENFORCE(it_end != it->end(), "error in parsing lod info");
    new_lod->emplace_back(it_begin, it_end + 1);
  }
  return new_lod;
}

}  // namespace details

}  // namespace framework
}  // namespace paddle
