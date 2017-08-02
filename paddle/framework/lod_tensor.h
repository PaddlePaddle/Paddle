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
#if (!PADDLE_ONLY_CPU)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "paddle/framework/ddim.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

/*
 * LODTensor (Level of details Tensor)
 * see https://en.wikipedia.org/wiki/Level_of_details for reference.
 */
class LODTensor {
 public:
// level_t save offsets of each unit.
#ifdef PADDLE_ONLY_CPU
  using level_t = std::vector<size_t>;
#else
  using level_t = thrust::device_vector<size_t>;
#endif
  // LOD stores offsets of each level of units, the largest units level first,
  // then the smaller units level. Each level_t stores the offsets of units in
  // Tesor.
  typedef std::vector<level_t> lod_t;

  LODTensor() {}
  LODTensor(std::shared_ptr<Tensor> tensor, std::shared_ptr<lod_t> lod) {
    Reset(tensor, lod);
  }

  void Reset(std::shared_ptr<Tensor> tensor, std::shared_ptr<lod_t> lod) {
    tensor_ = tensor;
    lod_start_pos_ = lod;
  }

  /*
   * Get a element from LOD.
   */
  size_t lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE(level < Levels(), "level [%d] out of range [%d]", level,
                   Levels());
    PADDLE_ENFORCE(elem < Elements(level),
                   "element begin [%d] out of range [%d]", elem,
                   Elements(level));
    return lod_start_pos_->at(level)[elem];
  }

  /*
   * Number of LODTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t Levels() const {
    return lod_start_pos_ ? lod_start_pos_->size() : 0UL;
  }
  /*
   * Number of elements in a level.
   */
  size_t Elements(size_t level = 0) const {
    PADDLE_ENFORCE(level < Levels(), "level [%d] out of range [%d]", level,
                   Levels());
    // the last offset is the end of last element
    return lod_start_pos_->at(level).size() - 1;
  }

  /*
   * Slice of levels[level_begin:level_end], with tensor copied.
   */
  template <typename T>
  LODTensor SliceCopied(size_t level_begin, size_t level_end,
                        const platform::Place &dst_place) const;

  /*
   * Slice of levels[level_begin:level_end], with tensor shared.
   */
  LODTensor SliceShared(size_t level_begin, size_t level_end) const;

  /*
   * Slice of elements of a level, [elem_begin: elem_end], with tensor copied.
   * @note: low performance in slice lod_start_pos_.
   */
  template <typename T>
  LODTensor SliceCopied(size_t level, size_t elem_begin, size_t elem_end,
                        const platform::Place &dst_place) const;

  /*
   * Slice of elements of a level, [elem_begin: elem_end], with tensor shared.
   * @note: low performance in slice lod_start_pos_.
   */
  template <typename T>
  LODTensor SliceShared(size_t level, size_t elem_begin, size_t elem_end) const;

  /*
   * Copy other's lod_start_pos_, to share LOD info.
   * @note: the LOD info should not be changed.
   */
  void ShareLOD(const LODTensor &other) {
    lod_start_pos_ = other.lod_start_pos_;
  }

  /*
   * Copy other's lod_start_pos_'s content, free to mutate.
   */
  void CopyLOD(const LODTensor &other) {
    lod_start_pos_ = std::make_shared<lod_t>(*other.lod_start_pos_);
  }
  /*
   * Determine whether LODTensor has a valid LOD info.
   */
  bool has_lod() const { return lod_start_pos_.get(); }
  std::shared_ptr<lod_t> const lod() const { return lod_start_pos_; }

  std::shared_ptr<Tensor> &tensor() const { return tensor_; }
  Tensor *raw_tensor() { return tensor_.get(); }

 private:
  mutable std::shared_ptr<lod_t> lod_start_pos_;
  mutable std::shared_ptr<Tensor> tensor_;
};

namespace details {

/*
 * Slice levels from LOD.
 *
 * @lod: LOD to slice.
 * @level_begin: level to begin slice.
 * @level_end: level to end slice.
 */
std::shared_ptr<LODTensor::lod_t> SliceLOD(const LODTensor::lod_t &lod,
                                           size_t level_begin,
                                           size_t level_end);

/*
 * Slice elements from a level of LOD.
 *
 * @lod: LOD to slice.
 * @level: which level to slice.
 * @elem_begin: element's index to begin slice.
 * @elem_end: element's index to end slice.
 */
std::shared_ptr<LODTensor::lod_t> SliceLOD(const LODTensor::lod_t &lod,
                                           size_t level, size_t elem_begin,
                                           size_t elem_end, bool tensor_shared);
}  // namespace details

template <typename T>
LODTensor LODTensor::SliceCopied(size_t level_begin, size_t level_end,
                                 const platform::Place &dst_place) const {
  PADDLE_ENFORCE(has_lod(), "has no LOD info, can't be sliced.");
  auto new_lod = details::SliceLOD(*lod_start_pos_, level_begin, level_end);
  auto new_tensor = std::make_shared<Tensor>();
  new_tensor->CopyFrom<T>(*tensor_, dst_place);

  return LODTensor(new_tensor, new_lod);
}

template <typename T>
LODTensor LODTensor::SliceShared(size_t level, size_t elem_begin,
                                 size_t elem_end) const {
  PADDLE_ENFORCE(has_lod(), "has no LOD info, can't be sliced.");
  PADDLE_ENFORCE(level < Levels(), "level [%d] out of range [%d]", level,
                 Levels());
  PADDLE_ENFORCE(elem_begin < Elements(level),
                 "element begin [%d] out of range [%d]", elem_begin,
                 Elements(level));
  PADDLE_ENFORCE(elem_end < Elements(level) + 1,
                 "element end [%d] out of range [%d]", elem_end,
                 Elements(level));

  auto new_lod = details::SliceLOD(*lod_start_pos_, level, elem_begin, elem_end,
                                   true /*tensor_shared*/);

  // slice elements just need to update LOD info, because offsets are not
  // changed, so the original tensor_ can be reused.
  return LODTensor(tensor_, new_lod);
}

template <typename T>
LODTensor LODTensor::SliceCopied(size_t level, size_t elem_begin,
                                 size_t elem_end,
                                 const platform::Place &dst_place) const {
  PADDLE_ENFORCE(has_lod(), "has no LOD info, can't be sliced.");
  PADDLE_ENFORCE(level < Levels(), "level [%d] out of range [%d]", level,
                 Levels());
  PADDLE_ENFORCE(elem_begin < Elements(level),
                 "element begin [%d] out of range [%d]", elem_begin,
                 Elements(level));
  PADDLE_ENFORCE(elem_end < Elements(level) + 1,
                 "element end [%d] out of range [%d]", elem_end,
                 Elements(level));

  auto new_lod = details::SliceLOD(*lod_start_pos_, level, elem_begin, elem_end,
                                   false /*tensor_shared*/);

  auto start_idx = new_lod->front().front();
  auto end_idx = new_lod->front().back() - 1 /*the next element's start*/;
  auto sliced_tensor = tensor_->Slice<T>(start_idx, end_idx);
  auto new_tensor = std::make_shared<Tensor>();
  new_tensor->template CopyFrom<T>(sliced_tensor, dst_place);

  return LODTensor(new_tensor, new_lod);
}

}  // namespace framework
}  // namespace paddle
