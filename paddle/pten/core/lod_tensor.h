/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/compat/mixed_vector.h"
#include "paddle/pten/core/base_tensor.h"

namespace pt {

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

/**
 * LoDTensor: compatible with LoDTensor in fluid and related operators.
 *
 * Note: LoDTensor (Level of details Tensor)
 * see https://en.wikipedia.org/wiki/Level_of_details for reference.
 */
class LoDTensor : public BaseTensor {
 public:
  LoDTensor() = delete;

  LoDTensor(const LoDTensor&) = delete;
  LoDTensor& operator=(const LoDTensor&) = delete;
  LoDTensor(LoDTensor&&) = delete;
  LoDTensor& operator=(LoDTensor&&) = delete;

  explicit LoDTensor(TensorMeta meta, const LoD& lod) : lod_(lod) {}

  void set_lod(const LoD& lod) { lod_ = lod; }

  const LoD& lod() const { return lod_; }

  LoD* mutable_lod() { return &lod_; }

 private:
  LoD lod_;
};

}  // namespace pt
