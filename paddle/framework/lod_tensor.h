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
#include "paddle/platform/place.h"

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
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
using LoD = std::vector<Vector<size_t>>;

std::ostream& operator<<(std::ostream& os, const LoD& lod);

/*
 * Slice levels from a LoD.
 * NOTE the lowest level should always be the absolute offsets of the underlying
 * tensor instances. So if higher layers are sliced without the lowest level,
 * the lower level of the sliced LoD will be transformed to the absolute offset.
 */
LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end);

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end);
/*
 * Transform an LoD from relative offsets to absolute offsets.
 */
LoD ToAbsOffset(const LoD& in);

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

  const LoD& lod() const { return lod_; }

  LoD* mutable_lod() { return &lod_; }

  /*
   * Get the start offset and end offset of an  element from LoD.
   */
  std::pair<size_t, size_t> lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE_LT(level, NumLevels());
    PADDLE_ENFORCE_LT(elem, NumElements(level));
    return std::make_pair((lod_)[level][elem], (lod_)[level][elem + 1]);
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
   * Get the number of instances in the underlying tensor in the `idx`-th
   * element.
   */
  size_t NumInstancesInElement(size_t level, size_t idx) const;

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

/*
 * Expand the `source` to fit the LoD of `lod`. For example, a `source`
 * LoDTensor is
 *  - LoD: [0, 2]
 *  - tensor: [a0, a1]
 * a `lod` is
 *  - LoD: [0 3 5]
 * returns a new LoDTensor
 *  - [a0 a0 a0 a1 a1]
 */
template <typename T>
LoDTensor LodExpand(const LoDTensor& source, const LoD& lod, size_t level,
                    const platform::Place& place) {
  LoD abs_lod = ToAbsOffset(lod);
  const auto& lod_level = lod[level];
  size_t num_instances = source.dims()[0];

  // new tensor
  LoDTensor tensor;
  tensor.set_lod(lod);
  auto dims = source.dims();
  dims[0] = lod_level.back();
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  PADDLE_ENFORCE_EQ(num_instances, lod_level.size() - 1);
  for (size_t ins = 0; ins < num_instances; ins++) {
    for (size_t elem = lod_level[ins]; elem < lod_level[ins + 1]; elem++) {
      tensor.Slice(elem, elem + 1)
          .CopyFrom(source.Slice(ins, ins + 1), platform::CPUPlace(),
                    platform::CPUDeviceContext());
    }
  }
  return tensor;
}

std::pair<LoD, std::pair<size_t, size_t>> GetSubLoDAndAbsoluteOffset(
    const LoD& lod, size_t start_idx, size_t end_idx, size_t start_level);

void AppendLoD(LoD* lod, const LoD& lod_length);

}  // namespace framework
}  // namespace paddle
