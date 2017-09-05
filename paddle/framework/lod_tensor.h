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
#ifndef PADDLE_ONLY_CPU
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "paddle/framework/ddim.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_ONLY_CPU
template <typename T>
using Vector = std::vector<T>;
#else
template <typename T>
using Vector = thrust::host_vector<T>;
#endif

using LOD = std::vector<Vector<size_t>>;

LOD SliceLevels(const LOD& in, size_t level_begin, size_t level_end);

LOD SliceInLevel(const LOD& in, size_t level, size_t elem_begin,
                 size_t elem_end);

bool operator==(const LOD& a, const LOD& b);

/*
 * LODTensor (Level of details Tensor)
 * see https://en.wikipedia.org/wiki/Level_of_details for reference.
 */
class LODTensor {
 public:
  LODTensor() {}
  LODTensor(const LOD& lod, Tensor* t) : lod_(lod), tensor_(t) {}

  void set_lod(const LOD& lod) { lod_ = lod; }

  void set_tensor(Tensor* tensor) { tensor_ = tensor; }

  Tensor& tensor() { return *tensor_; }

  LOD lod() { return lod_; }

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
    return (lod_)[level].size() - 1;
  }

  /*
   * Slice of levels[level_begin:level_end]
   */
  void SliceLevels(size_t level_begin, size_t level_end);

  /*
   * Slice of elements of a level, [elem_begin: elem_end]
   * @note: low performance in slice lod_.
   */
  void SliceInLevel(size_t level, size_t elem_begin, size_t elem_end);

 private:
  LOD lod_;
  Tensor* tensor_;  // not owned
};
}  // namespace framework
}  // namespace paddle
