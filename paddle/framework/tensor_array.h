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
#include <vector>

#include "paddle/framework/lod_tensor.h"

namespace paddle {
namespace framework {

/*
 * DyBatchSeqPosition stores indices of the basic element in tensor. It is used
 * after lod-tensor's re-assembling, its info can be used to recover the order
 * in original lod-tensor.
 */
struct DySeqMeta {
  DySeqMeta(size_t begin, size_t end, size_t ori_idx)
      : begin(begin), end(end), ori_idx(ori_idx) {}

  size_t begin;
  size_t end;  // not included
  size_t ori_idx;
};

using DySeqMetaBatch = std::vector<DySeqMeta>;

/*
 * Extract the indices of instances.
 */
std::vector<size_t> GenDyBatchIndice(const DySeqMetaBatch &metas, int batch_id);

/*
 * TensorArray is a C-array-like array of tensors, it is meant to be used with
 * dynamic iteration primitives such as while_loop. It is used to segment inputs
 * and store states in all time steps.
 *
 * By providing some methods similar to a C++ array, the difinition of some
 * state-based dynamic models such as RNN cound be more natural and highly
 * flexible.
 */
class TensorArray {
 public:
  using value_type = float;

  // max number of values allowed to store.
  const size_t MAX_SIZE{100000};

  /*
   * Read the value at location `index` in the `TensorArray`.
   */
  const LoDTensor &Read(size_t index) const;

  /*
   * Write value into the index of the TensorArray.
   */
  void Write(size_t index, const LoDTensor &value);

  /*
   * Write value into the index of the TensorArray, with memory shared.
   */
  void WriteShared(size_t index, const LoDTensor &value);

  /*
   * Recover the original LoD-arranged LoDTensor with the `values`, `level` and
   * `indice_map`.
   */
  LoDTensor Pack(size_t level, const DySeqMetaBatch &meta,
                 const LoD &lod) const;

  /*
   * Split LoDTensor in some `level` and write the generated batches to
   * `values`, if set `desend`, will sort by length in descending order else in
   * ascending order.
   */
  DySeqMetaBatch Unpack(const LoDTensor &source, int level, bool length_desend);

  /*
   * Pack an array of LoDTensors to a LoDTensor.
   */
  LoDTensor LodPack(size_t level) const;

  /*
   * Unpack a LoDTensor to an array of LoDTensors.
   */
  void LodUnpack(const LoDTensor &source, size_t level);

  /*
   * Pack the values into a tensor with rank one higher than each tensor in
   * values.
   */
  LoDTensor Stack() const;

  /*
   * Unstacks the given division of a rank-`R` tensor into rank-`(R-1)` tensors.
   */
  void Unstack(const LoDTensor &source) const;

  /*
   * Unstacks the given division of a rank-`R` tensor into rank-`(R-1)` tensors,
   * with memory of tensors shared.
   */
  void UnstackShared(const LoDTensor &source) const;

  /*
   * Return the number of values.
   */
  size_t size() const;

 protected:
  void Unstack(const LoDTensor &source, bool data_shared) const;

  LoDTensor LodPackTwo(const LoDTensor &pre, const LoDTensor &cur,
                       size_t level) const;

 private:
  mutable std::vector<LoDTensor> values_;
};  // class TensorArray

}  // namespace framework
}  // namespace paddle
