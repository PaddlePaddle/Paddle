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

#include "paddle/framework/tensor_array.h"

#include <glog/logging.h>
#include <algorithm>
#include <limits>

#include "paddle/framework/eigen.h"

namespace paddle {
namespace framework {

namespace detail {

/*
 * Offer an iterator over the length-sorted lod-tensor's top level. The top
 * level of a lod-tensor stores batch-size of sequences, each top-level sequence
 * may contains several lower-level sequences, sort top-level lod by the numbers
 * of lower-level sequences in descending order, so that during RNN's running,
 * the batch-size will keep decreasing, the short sentences will end at the tail
 * of each batch.
 *
 * Let's take a simple lod-tensor for example
 *
 *   |(0)       |(1)        top-level has two instances
 *   |||        |||||    lower-level
 *
 * sort by lower-level's length
 *
 *   |(1)       |(0)
 *   |||||      |||
 *
 * when RNN runs, it get 5 batches (equals the number of elements the longest
 * sequence has)
 *
 * |||||
 * |||
 *
 * the first three batches has two elements, the last two elements just has 1
 * element each.
 */
struct DynamicBatchUnpacker {
  using value_type = float;

  DynamicBatchUnpacker(const LoDTensor& source, size_t level,
                       bool descend = true)
      : source(&source), level(level) {
    BuildLengthSortedMeta(descend);
  }

  LoDTensor GetBatch(size_t index);

  std::vector<DySeqMeta> meta;

  LoDTensor const* source;
  size_t level;

 protected:
  void BuildLengthSortedMeta(bool descend);
};

LoDTensor PackDynamicBatch(const std::vector<LoDTensor>& source,
                           const std::vector<DySeqMeta>& meta, const LoD& lod,
                           size_t level);

std::vector<size_t> GenDyBatchIndice(const DySeqMetaBatch& meta, int batch_id) {
  // collect indice need to copy to the batch
  std::vector<size_t> indice;
  for (const auto& seq : meta) {
    size_t id = seq.begin + batch_id;
    if (id >= seq.end) break;
    indice.push_back(id);
  }
  return indice;
}

}  // namespace detail

const LoDTensor& TensorArray::Read(size_t index) const {
  PADDLE_ENFORCE_LE(index, MAX_SIZE, "index[%d] too large", index);
  if (index >= size()) {
    values_.resize(index + 1);
  }
  return values_[index];
}

void TensorArray::Write(size_t index, const LoDTensor& value) {
  PADDLE_ENFORCE_LE(index, MAX_SIZE, "index[%d] too large", index);

  if (index >= size()) {
    values_.resize(index + 1);
  }

  values_[index].set_lod(value.lod());
  values_[index].Resize(value.dims());
  values_[index].mutable_data<value_type>(value.place());
  values_[index].CopyFrom(value, value.place(), platform::CPUDeviceContext());
}

void TensorArray::WriteShared(size_t index, const LoDTensor& value) {
  PADDLE_ENFORCE_LE(index, MAX_SIZE, "index[%d] too large", index);
  if (index >= size()) {
    values_.resize(index + 1);
  }

  values_[index].set_lod(value.lod());
  values_[index].ShareDataWith(value);
}

LoDTensor TensorArray::Pack(size_t level, const std::vector<DySeqMeta>& meta,
                            const LoD& lod) const {
  return detail::PackDynamicBatch(values_, meta, lod, level);
}

DySeqMetaBatch TensorArray::Unpack(const LoDTensor& source, int level,
                                   bool length_desend) {
  detail::DynamicBatchUnpacker unpacker(source, level,
                                        length_desend /*descend*/);

  // find max length of all the sequences
  size_t max_length = 0;
  for (const auto& seq : unpacker.meta) {
    max_length = std::max(max_length, seq.end - seq.begin);
  }

  // write batches to values
  for (size_t batch_id = 0; batch_id < max_length; batch_id++) {
    Write(batch_id, unpacker.GetBatch(batch_id));
  }

  PADDLE_ENFORCE(!unpacker.meta.empty());
  return unpacker.meta;
}

LoDTensor TensorArray::LodPack(size_t level) const {
  PADDLE_ENFORCE_GT(size(), 0UL, "no time step exists");
  // the levels should be no less than 2
  LoDTensor merged;
  const LoDTensor *pre, *cur;
  pre = &Read(0);

  for (size_t step = 1; step < size(); step++) {
    cur = &Read(step);
    PADDLE_ENFORCE_GT(cur->NumLevels(), 0);
    PADDLE_ENFORCE_GT(pre->NumLevels(), 0);
    PADDLE_ENFORCE_EQ(pre->NumLevels(), cur->NumLevels());
    PADDLE_ENFORCE_EQ(pre->NumElements(level), cur->NumElements(level));

    merged = LodPackTwo(*pre, *cur, level);
    pre = &merged;
  }
  return merged;
}

/*
 * NOTE currently, only the lowest level supports packing.
 * The lowest LoD will be changed, while the relative offsets in levels above
 * stay unchanged.
 *
 * previous step : [0] [1] [3]
 * current step: [0 1 2] [2 3] []
 * packed to
 *   [0 0] [0 1] [0 2] [1 2] [1 3] [3]
 */
LoDTensor TensorArray::LodPackTwo(const LoDTensor& pre, const LoDTensor& cur,
                                  size_t level) const {
  PADDLE_ENFORCE_EQ(pre.NumLevels(), cur.NumLevels());
  PADDLE_ENFORCE_EQ(pre.NumLevels(), level + 1,
                    "Only the lowest LoD level supports pack temporarily.");
  // calculate the result tensor's shape first
  size_t num_instances = 0;
  for (size_t elem = 0; elem < pre.NumElements(level); elem++) {
    size_t prefix_size = pre.NumElements(level, elem);
    size_t num_candidates = cur.NumElements(level, elem);
    if (num_candidates > 0) {
      num_instances += num_candidates * (prefix_size + 1);
    } else {
      num_instances += prefix_size;
    }
  }

  auto res_dims = pre.dims();
  res_dims[0] = num_instances;
  LoDTensor result;
  result.Resize(res_dims);
  result.mutable_data<value_type>(cur.place());

  Vector<size_t> last_lod_level;
  // copy data
  size_t index = 0;
  last_lod_level.push_back(index);
  for (size_t elem = 0; elem < pre.NumElements(level); elem++) {
    size_t prefix_size = pre.NumElements(level, elem);
    size_t num_candidates = cur.NumElements(level, elem);

    // slice the prefix Tensor
    LoDTensor prefix = pre;
    prefix.ShrinkInLevel(level, elem, elem + 1);
    LoDTensor candidate = cur;
    if (num_candidates > 0) {
      candidate.ShrinkInLevel(level, elem, elem + 1);
    } else {  // just push prefix
      result.Slice(index, index + prefix_size)
          .CopyFrom(prefix, result.place(), platform::CPUDeviceContext());
      index += prefix_size;
      last_lod_level.push_back(index);
    }
    for (size_t candi = 0; candi < num_candidates; candi++) {
      // TODO(superjom) support GPU
      result.Slice(index, index + prefix_size)
          .CopyFrom(prefix, result.place(), platform::CPUDeviceContext());
      index += prefix_size;
      // copy candidate record
      result.Slice(index, index + 1)
          .CopyFrom(candidate.Slice(candi, candi + 1), result.place(),
                    platform::CPUDeviceContext());
      index++;
      last_lod_level.push_back(index);
    }
  }

  // update lod
  auto lod = cur.lod();
  lod.back() = last_lod_level;
  result.set_lod(lod);
  return result;
}

/*
 * source [0 1 2] [3 4] [5 6 7] will be transformd to a list of LoDTensors such
 * as
 * [0 3 5] [1 4 6] [2 7] with 1-level LoDs:
 * - [0 1 2 3]
 * - [0 1 2 3]
 * - [0 1 1 2], the [1,1) here means the second sequence is empty
 *
 * NOTE Unpack a LoDTensor in this approach may result in a big LoD.
 */
void TensorArray::LodUnpack(const LoDTensor& source, size_t level) {
  PADDLE_ENFORCE_EQ(level, source.NumLevels() - 1,
                    "only the lowest LoD level supports unpack.");
  const size_t non_empty_instances = source.dims()[0];
  size_t index = 0;
  Vector<size_t> lowest_lod_level;
  lowest_lod_level.push_back(index);

  for (size_t step = 0; step < non_empty_instances; step++) {
    size_t num_instances = 0;
    for (size_t id = 0; id < source.NumElements(level); id++) {
      auto instance = source;
      instance.ShrinkInLevel(level, id, id + 1);
      if (static_cast<size_t>(instance.dims()[0]) > step) {
        num_instances++;
        index++;
      }
      lowest_lod_level.push_back(index);
    }

    // create tensor for this time step
    LoDTensor tensor;
    auto dims = source.dims();
    dims[0] = num_instances;
    // set lod
    auto lod = source.lod();
    lod.back() = lowest_lod_level;
    tensor.set_lod(lod);

    index = 0;
    for (size_t id = 0; id < source.NumElements(level); id++) {
      auto instance = source;
      instance.ShrinkInLevel(level, id, id + 1);
      if (static_cast<size_t>(instance.dims()[0]) > step) {
        // copy this instance
        tensor.Slice(index, index + 1)
            .CopyFrom(instance.Slice(step, step + 1), tensor.place(),
                      platform::CPUDeviceContext());
        index++;
      }
    }
    Write(step, tensor);
  }
}

LoDTensor TensorArray::Stack() const {
  LoDTensor result;
  if (size() == 0) return result;

  const auto& first_dims = values_.front().dims();
  // check all the values have the same shape
  // TODO(superjom) check the same dtypes
  for (size_t idx = 1; idx < size(); idx++) {
    const auto& value_dims = values_[idx].dims();
    PADDLE_ENFORCE_EQ(first_dims, value_dims);
  }

  // copy
  auto result_dims = vectorize(first_dims);
  result_dims.insert(result_dims.begin(), size());
  result.Resize(make_ddim(result_dims));
  result.mutable_data<value_type>(platform::CPUPlace());

  for (size_t idx = 0; idx < size(); idx++) {
    result.Slice(idx, idx + 1)
        .CopyFrom(Read(idx), platform::CPUPlace(),
                  platform::CPUDeviceContext());
  }
  return result;
}

void TensorArray::Unstack(const LoDTensor& source) const {
  Unstack(source, false /*data_shared*/);
}

void TensorArray::UnstackShared(const LoDTensor& source) const {
  Unstack(source, true /*data_shared*/);
}

void TensorArray::Unstack(const LoDTensor& source, bool data_shared) const {
  size_t first_dim = source.dims()[0];
  DDim value_dims = slice_ddim(source.dims(), 1, source.dims().size());
  PADDLE_ENFORCE_GT(first_dim, 0,
                    "source should have some data to be unstacked");

  values_.resize(first_dim);

  for (size_t elem = 0; elem < first_dim; elem++) {
    // create a new value
    auto& value = values_[elem];
    if (data_shared) {
      // share memory
      value.ShareDataWith(source.Slice(elem, elem + 1));
    } else {
      // copy
      value.Resize(value_dims);
      value.CopyFrom(source.Slice(elem, elem + 1), platform::CPUPlace(),
                     platform::CPUDeviceContext());
    }
  }
}

size_t TensorArray::size() const { return values_.size(); }

namespace detail {

void DynamicBatchUnpacker::BuildLengthSortedMeta(bool descend) {
  PADDLE_ENFORCE(meta.empty(), "duplicate build meta");
  // collect meta for each sequence in some level
  auto lod = SliceLevels(source->lod(), level, level + 1)[0];

  for (size_t seq_id = 0; seq_id < lod.size() - 1; seq_id++) {
    DySeqMeta seq_meta({lod[seq_id], lod[seq_id + 1], seq_id});
    meta.push_back(seq_meta);
  }

  PADDLE_ENFORCE_GT(meta.size(), 0, "meta is empty");

  // sort by length
  sort(meta.begin(), meta.end(),
       [descend](const DySeqMeta& a, const DySeqMeta& b) {
         bool a_ge_b = (a.end - a.begin) > (b.end - b.begin);
         return descend ? a_ge_b : !a_ge_b;
       });
}

LoDTensor DynamicBatchUnpacker::GetBatch(size_t index) {
  PADDLE_ENFORCE(!meta.empty(), "should build meta first");
  LoDTensor result;

  auto indice = detail::GenDyBatchIndice(meta, index);
  PADDLE_ENFORCE(!indice.empty(), "invalid batch at %d", index);

  // copy the indice of records in LoDTensor
  auto record_dims = slice_ddim(source->dims(), 1, source->dims().size());
  auto record_dims_vec = vectorize(record_dims);
  record_dims_vec.insert(record_dims_vec.begin(), indice.size());
  result.Resize(make_ddim(record_dims_vec));
  result.mutable_data<value_type>(platform::CPUPlace());

  for (size_t i = 0; i < indice.size(); i++) {
    auto index = indice[i];
    auto target = result.Slice(i, i + 1);
    auto slice = source->Slice(index, index + 1);

    target.CopyFrom(slice, platform::CPUPlace(), platform::CPUDeviceContext());
  }

  return result;
}

// TODO(supejom) to cache lod if reasonable
LoDTensor PackDynamicBatch(const std::vector<LoDTensor>& source,
                           const std::vector<DySeqMeta>& meta, const LoD& lod,
                           size_t level) {
  PADDLE_ENFORCE(!source.empty());
  PADDLE_ENFORCE(!meta.empty());
  PADDLE_ENFORCE(!lod.empty());

  LoDTensor result;

  // init result space
  auto record_dims = slice_ddim(source[0].dims(), 1, source[0].dims().size());
  auto record_dims_vec = vectorize(record_dims);
  auto height = lod[level].back();
  record_dims_vec.insert(record_dims_vec.begin(), height);
  result.Resize(make_ddim(record_dims_vec));
  result.mutable_data<float>(platform::CPUPlace());

  for (size_t batch_id = 0; batch_id < source.size(); batch_id++) {
    for (size_t seq_id = 0; seq_id < meta.size(); seq_id++) {
      const auto& seq_meta = meta[seq_id];
      // source is source[batch_id][seq_id]
      // target is result[index]
      auto index = seq_meta.begin + batch_id;
      if (index >= seq_meta.end) break;
      auto source_ = source[batch_id].Slice(seq_id, seq_id + 1);
      auto target = result.Slice(index, index + 1);
      target.CopyFrom(source_, platform::CPUPlace(),
                      platform::CPUDeviceContext());
    }
  }

  result.set_lod(lod);
  return result;
}

}  // namespace detail

}  // namespace framework
}  // namespace paddle
