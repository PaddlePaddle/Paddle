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
#ifdef PADDLE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

#include <glog/logging.h>
#include "paddle/framework/ddim.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

#ifndef PADDLE_WITH_CUDA
template <typename T>
using Vector = std::vector<T>;
#else
template <typename T>
using Vector = thrust::host_vector<
    T, thrust::system::cuda::experimental::pinned_allocator<T>>;
#endif

/*
 * 3-level LoD stores
 *
 * 0 10 20
 * 0 5 10 15 20
 * 0 2 5 7 10 12 15 20
 *
 * - in a level, each element indicates offset in the underlying Tensor
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 */
using LoD = std::vector<Vector<size_t>>;

LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end);

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end);

bool operator==(const LoD& a, const LoD& b);

/*
 * LoDTensor (Level of details Tensor)
 * see https://en.wikipedia.org/wiki/Level_of_details for reference.
 */
class LoDTensor : public Tensor {
 public:
  LoDTensor() {}

  explicit LoDTensor(const LoD& lod) : lod_(lod) {}

  void set_lod(const LoD& lod) { lod_ = lod; }

  LoD lod() const { return lod_; }

  /*
   * Get a element from LoD.
   */
  size_t lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE_LT(level, NumLevels());
    PADDLE_ENFORCE_LT(elem, NumElements(level));
    return (lod_)[level][elem];
  }

  /*
   * Number of LoDTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t NumLevels() const { return lod_.size(); }
  /*
   * Number of elements in a level.
   */
  size_t NumElements(size_t level = 0) const {
    PADDLE_ENFORCE_LT(level, NumLevels());
    // the last offset is the end of last element
    return (lod_)[level].size() - 1;
  }

  /*
   * Number of lower-level elements.
   * For example, a 2-level lod-tensor
   *
   * 0-th level   |   |
   * 1-th level   ||  |||
   *
   * NumElements(0, 0) get 2
   * NumElements(0, 1) get 3
   */
  size_t NumElements(size_t level, size_t idx) const;

  /*
   * Shrink levels[level_begin:level_end]
   */
  void ShrinkLevels(size_t level_begin, size_t level_end);

  /*
   * Shrink elements of a level, [elem_begin: elem_end]
   * @note: low performance in slice lod_.
   */
  void ShrinkInLevel(size_t level, size_t elem_begin, size_t elem_end);

 private:
  LoD lod_;
};
}  // namespace framework
}  // namespace paddle
