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
class LODTensor : public Tensor {
 public:
// Level save offsets of each unit.
#ifdef PADDLE_ONLY_CPU
  using Level = std::vector<size_t>;
#else
  using Level = thrust::device_vector<size_t>;
#endif
  // LOD stores offsets of each level of units, the largest units level first,
  // then the smaller units level. Each Level stores the offsets of units in
  // Tesor.
  typedef std::vector<Level> LOD;

  LODTensor() {}
  LODTensor(const std::shared_ptr<Tensor> &tensor,
            const std::shared_ptr<LOD> &lod) {
    Reset(tensor, lod);
  }

  void Reset(const std::shared_ptr<Tensor> &tensor,
             const std::shared_ptr<LOD> &lod) {
    tensor_ = tensor;
    lod_start_pos_ = lod;
  }

  /*
   * Get a element from LOD.
   */
  size_t lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                   NumLevels());
    PADDLE_ENFORCE(elem < NumElements(level),
                   "element begin [%d] out of range [%d]", elem,
                   NumElements(level));
    return (*lod_start_pos_)[level][elem];
  }

  /*
   * Number of LODTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t NumLevels() const {
    return lod_start_pos_ ? lod_start_pos_->size() : 0UL;
  }
  /*
   * Number of elements in a level.
   */
  size_t NumElements(size_t level = 0) const {
    PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                   NumLevels());
    // the last offset is the end of last element
    return lod_start_pos_->at(level).size() - 1;
  }

  /*
   * Slice of levels[level_begin:level_end], with tensor shared.
   */
  LODTensor SliceLevels(size_t level_begin, size_t level_end) const;

  /*
   * Slice of elements of a level, [elem_begin: elem_end], with tensor shared.
   * @note: low performance in slice lod_start_pos_.
   */
  LODTensor SliceInLevel(size_t level, size_t elem_begin,
                         size_t elem_end) const;

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
    lod_start_pos_ = std::make_shared<LOD>(*other.lod_start_pos_);
  }
  /*
   * Determine whether LODTensor has a valid LOD info.
   */
  bool HasLOD() const { return bool(lod_start_pos_); }
  LOD *lod() const { return lod_start_pos_.get(); }

  std::shared_ptr<Tensor> &tensor() { return tensor_; }
  Tensor *raw_tensor() { return tensor_.get(); }

 private:
  std::shared_ptr<LOD> lod_start_pos_;
  std::shared_ptr<Tensor> tensor_;
};

}  // namespace framework
}  // namespace paddle
