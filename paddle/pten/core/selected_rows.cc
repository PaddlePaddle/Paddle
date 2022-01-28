/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/selected_rows.h"
#include "paddle/pten/core/utils/data_type.h"

namespace pten {

struct ReAllocateVisitor {
  ReAllocateVisitor(const pten::framework::DDim& dims,
                    pten::DenseTensor* tensor)
      : dims_(dims), tensor_(tensor) {}

  template <typename T>
  void operator()() const {
    pten::DenseTensor cpu_tensor;
    paddle::platform::CPUPlace cpu;
    T* ptr = cpu_tensor.mutable_data<T>(dims_, cpu);
    const T* old_ptr =
        tensor_->memory_size() == 0 ? nullptr : tensor_->data<T>();
    if (old_ptr != nullptr) {
      std::copy(old_ptr, old_ptr + tensor_->numel(), ptr);
    }
    tensor_->ShareDataWith(cpu_tensor);
  }

  pten::framework::DDim dims_;
  pten::DenseTensor* tensor_;
};

struct TensorCopyVisitor {
  TensorCopyVisitor(pten::DenseTensor* dst,
                    int64_t dst_offset,
                    const pten::DenseTensor src,
                    int64_t src_offset,
                    int64_t size)
      : dst_(dst),
        dst_offset_(dst_offset),
        src_(src),
        src_offset_(src_offset),
        size_(size) {}

  template <typename T>
  void apply() const {
    // TODO(Yancey1989): support other place
    paddle::platform::CPUPlace cpu;
    paddle::memory::Copy(cpu,
                         dst_->mutable_data<T>(cpu) + dst_offset_,
                         cpu,
                         src_.data<T>() + src_offset_,
                         size_ * sizeof(T));
  }

  pten::DenseTensor* dst_;
  int64_t dst_offset_;
  pten::DenseTensor src_;
  int64_t src_offset_;
  int64_t size_;
};

struct TensorFillVisitor {
  TensorFillVisitor(pten::DenseTensor* dst,
                    int64_t dst_offset,
                    int64_t size,
                    float value)
      : dst_(dst), dst_offset_(dst_offset), size_(size) {}

  template <typename T>
  void apply() const {
    // TODO(qiao): support other place
    paddle::platform::CPUPlace cpu;
    auto* tensor_data = dst_->mutable_data<T>(cpu);
    auto* start = tensor_data + dst_offset_;
    auto* end = start + size_;
    std::fill(start, end, static_cast<T>(0.0));
  }

  pten::DenseTensor* dst_;
  int64_t dst_offset_;
  int64_t size_;
};

void* SelectedRows::AllocateFrom(Allocator* allocator,
                                 DataType dtype,
                                 size_t requested_size) {
  return value_->AllocateFrom(allocator, dtype, requested_size);
}

bool SelectedRows::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

int64_t SelectedRows::AutoGrownIndex(int64_t key,
                                     bool auto_grown,
                                     bool is_test) {
  if (is_test) {
    auto iter = id_to_index_.find(key);
    if (iter == id_to_index_.end()) {
      return -1;
    } else {
      return iter->second;
    }
  }

  rwlock_->RDLock();
  auto iter = id_to_index_.find(key);
  if (iter == id_to_index_.end()) {
    rwlock_->UNLock();
    PADDLE_ENFORCE_EQ(auto_grown,
                      true,
                      paddle::platform::errors::NotFound(
                          "Input key(%lld) is not found.", key));
    rwlock_->WRLock();
    auto map_size = id_to_index_.size();
    auto vector_size = rows_.size();
    if (map_size != vector_size) {
      rwlock_->UNLock();
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Row map size(%zu) should be equal to rows size(%zu).",
          map_size,
          vector_size));
    }
    auto write_iter = id_to_index_.find(key);
    if (write_iter == id_to_index_.end()) {
      int row_num = rows_.size();
      if (row_num == value_->dims()[0]) {
        rwlock_->UNLock();
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Selected rows is full, then length exceed the length of first "
            "dimension (%d).",
            row_num));
      }
      // key logic to put a key into id_to_index_
      rows_.push_back(key);
      auto index = static_cast<int64_t>(rows_.size() - 1);
      id_to_index_[key] = index;
      rwlock_->UNLock();
      return index;
    } else {
      auto index = write_iter->second;
      rwlock_->UNLock();
      return index;
    }
  } else {
    auto index = iter->second;
    rwlock_->UNLock();
    return index;
  }
}

void SelectedRows::SyncIndex() {
  rwlock_->WRLock();
  id_to_index_.clear();
  for (size_t i = 0; i < rows_.size(); ++i) {
    id_to_index_[rows_[i]] = i;
  }
  rwlock_->UNLock();
}

void SelectedRows::Get(const pten::DenseTensor& ids,
                       pten::DenseTensor* value,
                       bool auto_grown,
                       bool is_test) {
  PADDLE_ENFORCE_EQ(value->IsInitialized(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The value tensor is not initialized."));
  if (ids.numel() == 0) {
    VLOG(3) << "keys is empty, please check data!";
  } else {
    int64_t value_width = value_->numel() / value_->dims()[0];
    PADDLE_ENFORCE_EQ(
        value_width,
        value->numel() / value->dims()[0],
        paddle::platform::errors::InvalidArgument(
            "Output tensor should have the same shape with table "
            "except the first dimmension, excepted value width not counting "
            "the first dimension is %d, actual value width is %d.",
            value_width,
            value->numel() / value->dims()[0]));
    for (int i = 0; i < ids.numel(); ++i) {
      auto id = ids.data<int64_t>()[i];
      int64_t index = AutoGrownIndex(id, auto_grown, is_test);
      if (index < 0) {
        VLOG(5) << "id " << id << " not in the table, return 0";
        pten::VisitDataType(
            value_->dtype(),
            TensorFillVisitor(value, i * value_width, value_width, 0.0));
      } else {
        pten::VisitDataType(value_->dtype(),
                            TensorCopyVisitor(value,
                                              i * value_width,
                                              *value_.get(),
                                              index * value_width,
                                              value_width));
      }
    }
  }
}
}  // namespace pten
