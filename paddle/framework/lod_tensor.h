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
#if !defined(PADDLE_ONLY_CPU)
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
class LODTensor : public Tensor {
 public:
// Level save offsets of each unit.
#ifdef PADDLE_ONLY_CPU
  template <typename T>
  using Vector = std::vector<T>;
#else
  template <typename T>
  using Vector = thrust::host_vector<T>;
#endif
  // LoD stores offsets of each level of units, the largest units level first,
  // then the smaller units level. Each Level stores the offsets of units in
  // Tesor.
  class LOD : public std::vector<Vector<size_t>> {
   public:
    LOD SliceLevels(size_t level_begin, size_t level_end) const;
    LOD SliceInLevel(size_t level, size_t elem_begin, size_t elem_end) const;
  };

  LODTensor() {}
  explicit LODTensor(const LOD &lod) : lod_(lod) {}

  virtual Tensor *Clone() const { return new LODTensor(lod_); }

  /*
   * Get a element from LOD.
   */
  size_t lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                   NumLevels());
    PADDLE_ENFORCE(elem < NumElements(level),
                   "element begin [%d] out of range [%d]", elem,
                   NumElements(level));
    return (lod_)[level][elem];
  }

  /*
   * Number of LODTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t NumLevels() const { return lod_.size(); }
  /*
   * Number of elements in a level.
   */
  size_t NumElements(size_t level = 0) const {
    PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                   NumLevels());
    // the last offset is the end of last element
    return lod_[level].size() - 1;
  }

  /*
   * Slice of levels[level_begin:level_end], with tensor shared.
   */
  template <typename T>
  LODTensor SliceLevels(size_t level_begin, size_t level_end) const;

  /*
   * Slice of elements of a level, [elem_begin: elem_end], with tensor shared.
   * @note: low performance in slice lod_.
   */
  template <typename T>
  LODTensor SliceInLevel(size_t level, size_t elem_begin,
                         size_t elem_end) const;

  /*
   * Copy other's lod_'s content, free to mutate.
   */
  void CopyLOD(const LODTensor &other) { lod_ = other.lod_; }
  /*
   * Determine whether LODTensor has a valid LOD info.
   */
  const LOD &lod() const { return lod_; }
  LOD *mutable_lod() { return &lod_; }

  virtual ~LODTensor() {}

 private:
  LOD lod_;
};

bool operator==(const LODTensor::LOD &a, const LODTensor::LOD &b);

template <typename T>
LODTensor LODTensor::SliceLevels(size_t level_begin, size_t level_end) const {
  auto new_lod = lod_.SliceLevels(level_begin, level_end);
  // slice levels just need to update LOD info, each level will contains the
  // whole tensor_, so no need to modify tensor_.
  LODTensor new_tensor(new_lod);
  new_tensor.ShareDataWith<T>(*this);
  return new_tensor;
}

template <typename T>
LODTensor LODTensor::SliceInLevel(size_t level, size_t elem_begin,
                                  size_t elem_end) const {
  PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                 NumLevels());
  PADDLE_ENFORCE(elem_begin < NumElements(level),
                 "element begin [%d] out of range [%d]", elem_begin,
                 NumElements(level));
  PADDLE_ENFORCE(elem_end < NumElements(level) + 1,
                 "element end [%d] out of range [%d]", elem_end,
                 NumElements(level));

  auto new_lod = lod_.SliceInLevel(level, elem_begin, elem_end);

  // slice elements just need to update LOD info, because offsets are not
  // changed, so the original tensor_ can be reused.
  LODTensor new_tensor(new_lod);
  new_tensor.ShareDataWith<T>(*this);
  return new_tensor;
}

}  // namespace framework
}  // namespace paddle
