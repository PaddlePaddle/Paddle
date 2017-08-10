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

#include "paddle/framework/details/lod_tensor.h"

namespace paddle {
namespace framework {

template <typename T>
LODTensor LODTensor::SliceCopied(size_t level_begin, size_t level_end,
                                 const platform::Place &dst_place) const {
  PADDLE_ENFORCE(HasLOD(), "has no LOD info, can't be sliced.");
  auto new_lod = details::SliceLOD(*lod_start_pos_, level_begin, level_end);
  auto new_tensor = std::make_shared<Tensor>();
  new_tensor->CopyFrom<T>(*tensor_, dst_place);

  return LODTensor(new_tensor, new_lod);
}

template <typename T>
LODTensor LODTensor::SliceCopied(size_t level, size_t elem_begin,
                                 size_t elem_end,
                                 const platform::Place &dst_place) const {
  PADDLE_ENFORCE(HasLOD(), "has no LOD info, can't be sliced.");
  PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                 NumLevels());
  PADDLE_ENFORCE(elem_begin < NumElements(level),
                 "element begin [%d] out of range [%d]", elem_begin,
                 NumElements(level));
  PADDLE_ENFORCE(elem_end < NumElements(level) + 1,
                 "element end [%d] out of range [%d]", elem_end,
                 NumElements(level));

  auto new_lod = details::SliceLOD(*lod_start_pos_, level, elem_begin, elem_end,
                                   false /*tensor_shared*/);

  auto start_idx = new_lod->front().front();
  auto end_idx = new_lod->front().back() - 1 /*the next element's start*/;
  auto sliced_tensor = tensor_->Slice<T>(start_idx, end_idx);
  auto new_tensor = std::make_shared<Tensor>();
  new_tensor->CopyFrom<T>(sliced_tensor, dst_place);

  return LODTensor(new_tensor, new_lod);
}

}  // namespace framework
}  // namespace paddle
