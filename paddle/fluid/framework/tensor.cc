/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

DECLARE_bool(use_stream_safe_cuda_allocator);

namespace paddle {
namespace memory {
namespace allocation {
class Allocation;
}  // namespace allocation
}  // namespace memory
}  // namespace paddle

namespace paddle {
namespace framework {
extern size_t SizeOfType(proto::VarType::Type type);
void Tensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(holder_, platform::errors::PreconditionNotMet(
                                       "Tensor holds no memory. "
                                       "Call Tensor::mutable_data firstly."));
  size_t size = numel() * SizeOfType(type());

  PADDLE_ENFORCE_LE(
      size, memory_size(),
      platform::errors::PreconditionNotMet(
          "Tensor's dimension is out of bound."
          "Tensor's dimension must be equal or less than the size of its "
          "memory."
          "But received  Tensor's dimension is d%, memory's size is %d.",
          size, memory_size()));
}

Tensor::Tensor(const proto::VarType::Type& dtype)
    : type_(dtype),
      offset_(0),
      inplace_version_counter_(std::make_shared<TensorInplaceVersion>(0)) {}

size_t Tensor::memory_size() const {
  return holder_ == nullptr ? 0UL : holder_->size() - offset_;
}

void* Tensor::mutable_data(const platform::Place& place,
                           proto::VarType::Type type, size_t requested_size) {
  type_ = type;
  PADDLE_ENFORCE_GE(
      numel(), 0,
      platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(), "] now"));
  size_t size = numel() * SizeOfType(type);
  if (requested_size && (requested_size > size)) {
    size = requested_size;
  }
  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    // Reset holder first before re-allocate to save memory
    holder_.reset();
    holder_ = memory::AllocShared(place, size);
    offset_ = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
}

void* Tensor::mutable_data(const platform::Place& place,
                           size_t requested_size) {
  PADDLE_ENFORCE_NOT_NULL(this->holder_, platform::errors::PreconditionNotMet(
                                             "The tensor is not initialized."));
  return mutable_data(place, type_, requested_size);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void* Tensor::mutable_data(const platform::CUDAPlace& place,
                           proto::VarType::Type type,
                           const gpuStream_t& stream) {
  if (!FLAGS_use_stream_safe_cuda_allocator) {
    return mutable_data(place, type);
  }

  type_ = type;
  PADDLE_ENFORCE_GE(
      numel(), 0,
      platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(), "] now"));
  size_t size = numel() * SizeOfType(type);

  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    holder_.reset();
    holder_ = memory::AllocShared(place, size, stream);
    offset_ = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
}
#endif

Tensor& Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size();
  *this = src;
  return *this;
}
Tensor& Tensor::ShareInplaceVersionCounterWith(const Tensor& src) {
  PADDLE_ENFORCE_NOT_NULL(
      inplace_version_counter_,
      platform::errors::PreconditionNotMet(
          "Tensor does not hold inplace_version_counter_."));

  inplace_version_counter_ = src.inplace_version_counter_;
  return *this;
}

Tensor Tensor::Slice(int64_t begin_idx, int64_t end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(
      begin_idx, 0,
      platform::errors::OutOfRange("The start row index must be greater than 0."
                                   "But received the start index is d%.",
                                   begin_idx));
  PADDLE_ENFORCE_LE(
      end_idx, dims_[0],
      platform::errors::OutOfRange("The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_idx, end_idx,
      platform::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx, end_idx));

  if (dims_[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / dims_[0];
    Tensor dst;
    dst.holder_ = holder_;
    dst.set_layout(layout_);
    dst.type_ = type_;
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * SizeOfType(type());
    return dst;
  }
}

std::vector<Tensor> Tensor::Split(int64_t split_size, int64_t axis) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(dims_.size(), 0,
                    platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));
  PADDLE_ENFORCE_GE(
      split_size, 0,
      platform::errors::OutOfRange(
          "split expects split_size be non-negative, but got split_size is %d",
          split_size));
  int64_t numel_size = dims_[axis];

  int64_t num_splits = 1;
  if (split_size != 0) {
    num_splits =
        std::max<int64_t>((numel_size + split_size - 1) / split_size, 1);
  }

  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - numel_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    int64_t length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = Slice(i * split_size, i * split_size + length);
  }
  return splits;
}

std::vector<Tensor> Tensor::Chunk(int64_t chunks, int64_t axis) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(dims_.size(), 0,
                    platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));
  PADDLE_ENFORCE_GE(
      chunks, 0,
      platform::errors::OutOfRange(
          "chunks expects to be greater than 0, but got chunks is %d", chunks));

  int64_t numel_size = dims_[axis];
  int64_t split_size = (numel_size + chunks - 1) / chunks;
  return Split(split_size, axis);
}

Tensor& Tensor::Resize(const DDim& dims) {
  dims_ = dims;
  return *this;
}

const DDim& Tensor::dims() const { return dims_; }

int64_t Tensor::numel() const { return product(dims_); }

void Tensor::ResetHolder(std::shared_ptr<memory::Allocation> holder) {
  PADDLE_ENFORCE_EQ(
      offset_, 0,
      platform::errors::Fatal(
          "Only the offset is supported to zero when the holder is reset."));
  if (holder_) {
    PADDLE_ENFORCE_LE(
        numel() * SizeOfType(type()) + offset_, holder->size(),
        paddle::platform::errors::InvalidArgument(
            "The size of Holder is not enough to store the Tensor."));
  }
  holder_ = holder;
}

void Tensor::ResetHolderWithType(std::shared_ptr<memory::Allocation> holder,
                                 const proto::VarType::Type& type) {
  type_ = type;
  ResetHolder(holder);
}

void Tensor::set_type(const proto::VarType::Type& type) { type_ = type; }

/* ---------------------------------------- */
/* -------------- Tensor --------------- */
/* ---------------------------------------- */
using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD& lod, size_t start_idx,
                                        size_t end_idx, size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_ENFORCE_LE(start_idx, end_idx,
                      platform::errors::InvalidArgument(
                          "The start index should be less than the end index, "
                          "but received start index is %d, end index is %d.",
                          start_idx, end_idx));
    PADDLE_ENFORCE_LT(
        end_idx, lod[level_idx].size(),
        platform::errors::InvalidArgument(
            "The end index should be less than the LoD level size, but "
            "received end index is %d, LoD level size is %d.",
            end_idx, lod[level_idx].size()));
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

std::vector<Tensor> Tensor::SplitLoDTensor(
    const std::vector<platform::Place> places) const {
  PADDLE_ENFORCE_GT(places.size(), 0,
                    platform::errors::InvalidArgument(
                        "Place number cannot be empty when splitting."));
  check_memory_size();
  size_t batch_size =
      lod().empty() ? static_cast<size_t>(dims()[0]) : lod()[0].size() - 1;

  // if batch_size is 0, just return #places.size() copys of empty
  // tensors.
  if (batch_size == 0) {
    std::vector<Tensor> empty_results;
    empty_results.reserve(places.size());
    for (size_t i = 0; i < places.size(); ++i) {
      Tensor dst;
      dst.Resize(dims());
      dst.mutable_data(places[i], type());
      if (!lod().empty()) {
        dst.set_lod(lod());
      }
      empty_results.emplace_back(std::move(dst));
    }
    return empty_results;
  }

  auto step_width = (batch_size + places.size() - 1) / places.size();
  auto result_size = (batch_size + step_width - 1) / step_width;
  std::vector<Tensor> results;
  results.reserve(result_size);

  for (size_t i = 0; i < result_size; ++i) {
    auto begin = i * step_width;
    auto end = std::min<size_t>((i + 1) * step_width, batch_size);
    PADDLE_ENFORCE_LT(begin, end,
                      platform::errors::InvalidArgument(
                          "The begin index must be less than the end index, "
                          "but received begin index is %d, end index is %d.",
                          begin, end));

    Tensor dst;
    if (lod().empty()) {
      auto src = Slice(begin, end);
      auto& dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);
    } else {
      auto lod_and_offset = GetSubLoDAndAbsoluteOffset(lod(), begin, end, 0);

      auto& offset = lod_and_offset.second;
      auto src = Slice(offset.first, offset.second);
      auto& dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);

      LoD my_lod;
      for (auto& l : lod_and_offset.first) {
        std::vector<size_t> v{0};
        for (auto& ll : l) {
          v.push_back(ll + v.back());
        }
        my_lod.emplace_back(v);
      }
      dst.set_lod(my_lod);
    }
    results.emplace_back(std::move(dst));
  }

  return results;
}

void Tensor::MergeLoDTensor(const std::vector<const Tensor*>& lod_tensors,
                            platform::Place dst_place) {
  PADDLE_ENFORCE_EQ(lod_tensors.empty(), false,
                    platform::errors::InvalidArgument(
                        "The LoDTensors to be merged are empty."));
  framework::DDim new_dim = lod_tensors[0]->dims();
  proto::VarType::Type new_type = proto::VarType::FP32;
  framework::DataLayout new_layout = lod_tensors[0]->layout();
  for (auto* t : lod_tensors) {
    if (t->numel() && t->IsInitialized()) {
      new_dim = t->dims();
      new_type = t->type();
      new_layout = t->layout();
      break;
    }
  }

  LoD new_lod = lod_tensors[0]->lod();

  for (size_t i = 1; i < lod_tensors.size(); ++i) {
    auto* t = lod_tensors[i];
    if (t->numel() && t->IsInitialized()) {
      PADDLE_ENFORCE_EQ(
          new_type, t->type(),
          platform::errors::InvalidArgument(
              "Tensor data type does not match, expected type is %s, actual "
              "type is %s.",
              DataTypeToString(new_type), DataTypeToString(t->type())));
      PADDLE_ENFORCE_EQ(
          new_layout, t->layout(),
          platform::errors::InvalidArgument(
              "Tensor layout does not match, expected layout is %s, "
              "actual layout is %s.",
              DataLayoutToString(new_layout), DataLayoutToString(t->layout())));
      PADDLE_ENFORCE_EQ(
          framework::product(new_dim) / new_dim[0],
          framework::product(t->dims()) / t->dims()[0],
          platform::errors::InvalidArgument(
              "Tensor dimension does not match, all dimensions except the "
              "first dimension need to be equal,"
              "but expected dimension is %s, actual dimension is %s.",
              new_dim, t->dims()));
      new_dim[0] += t->dims()[0];
    }

    auto& lod = t->lod();
    PADDLE_ENFORCE_EQ(new_lod.size(), lod.size(),
                      platform::errors::InvalidArgument(
                          "The LoD information of Tensor does not match"));

    for (size_t j = 0; j < lod.size(); ++j) {
      auto& sub_lod = new_lod[j];
      size_t offset = sub_lod.back();
      for (size_t k = 1; k < lod[j].size(); ++k) {
        sub_lod.push_back(lod[j][k] + offset);
      }
    }
  }
  Resize(new_dim);
  set_layout(new_layout);
  set_lod(new_lod);
  mutable_data(dst_place, new_type);

  int begin = 0;
  for (auto* src : lod_tensors) {
    int end = begin + src->dims()[0];
    if (end == begin) {
      continue;
    }
    auto dst = Slice(begin, end);
    framework::TensorCopy(*src, dst_place, &dst);
    begin = end;
  }
}

}  // namespace framework
}  // namespace paddle
