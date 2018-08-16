/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace framework {

struct ReAllocateVisitor {
  ReAllocateVisitor(const framework::DDim& dims, framework::Tensor* tensor)
      : dims_(dims), tensor_(tensor) {}

  template <typename T>
  void operator()() const {
    framework::Tensor cpu_tensor;
    platform::CPUPlace cpu;
    T* ptr = cpu_tensor.mutable_data<T>(dims_, cpu);
    const T* old_ptr =
        tensor_->memory_size() == 0 ? nullptr : tensor_->data<T>();
    if (old_ptr != nullptr) {
      std::copy(old_ptr, old_ptr + tensor_->numel(), ptr);
    }
    tensor_->ShareDataWith(cpu_tensor);
  }

  framework::DDim dims_;
  framework::Tensor* tensor_;
};

struct TensorCopyVisitor {
  TensorCopyVisitor(framework::Tensor* dst, int64_t dst_offset,
                    const framework::Tensor src, int64_t src_offset,
                    int64_t size)
      : dst_(dst),
        dst_offset_(dst_offset),
        src_(src),
        src_offset_(src_offset),
        size_(size) {}

  template <typename T>
  void operator()() const {
    // TODO(Yancey1989): support other place
    platform::CPUPlace cpu;
    memory::Copy(cpu, dst_->mutable_data<T>(cpu) + dst_offset_, cpu,
                 src_.data<T>() + src_offset_, size_ * sizeof(T));
  }

  framework::Tensor* dst_;
  int64_t dst_offset_;
  framework::Tensor src_;
  int64_t src_offset_;
  int64_t size_;
};

void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {
    // the 2st field, rows information
    auto& rows = selected_rows.rows();
    uint64_t size = rows.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char*>(&rows[i]), sizeof(rows[i]));
    }
  }
  {
    // the 3st field, the height of SelectedRows
    int64_t height = selected_rows.height();
    os.write(reinterpret_cast<const char*>(&height), sizeof(height));
  }
  // the 4st field, Tensor data
  TensorToStream(os, selected_rows.value(), dev_ctx);
}

void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
  }
  {
    // the 2st field, rows information
    uint64_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    auto& rows = *selected_rows->mutable_rows();
    rows.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      is.read(reinterpret_cast<char*>(&rows[i]), sizeof(int64_t));
    }
  }
  {
    // the 3st field, the height of the SelectedRows
    int64_t height;
    is.read(reinterpret_cast<char*>(&height), sizeof(int64_t));
    selected_rows->set_height(height);
  }
  // the 4st field, tensor which contains the data
  TensorFromStream(is, selected_rows->mutable_value(), dev_ctx);
}

bool SelectedRows::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

int64_t SelectedRows::AutoGrownIndex(int64_t key, bool auto_grown) {
  rwlock_->RDLock();
  auto iter = id_to_index_.find(key);
  if (iter == id_to_index_.end()) {
    rwlock_->UNLock();
    if (!auto_grown) {
      PADDLE_THROW("key %d not found", key);
    }
    rwlock_->WRLock();
    auto map_size = id_to_index_.size();
    auto vector_size = rows_.size();
    if (map_size != vector_size) {
      rwlock_->UNLock();
      PADDLE_THROW(
          "id_to_index_ size %d should have the same size with rows_ %d",
          map_size, vector_size);
    }
    auto write_iter = id_to_index_.find(key);
    if (write_iter == id_to_index_.end()) {
      size_t row_num = rows_.size();
      if (row_num == value_->dims()[0]) {
        rwlock_->UNLock();
        PADDLE_THROW("selected rows is full, then length exceed %d", row_num);
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

void SelectedRows::Get(const framework::Tensor& ids, framework::Tensor* value,
                       bool auto_grown) {
  PADDLE_ENFORCE(value->IsInitialized(),
                 "The value tensor should be initialized.");
  if (ids.numel() == 0) {
    VLOG(3) << "keys is empty, please check data!";
  } else {
    int64_t value_width = value_->numel() / value_->dims()[0];
    PADDLE_ENFORCE_EQ(value_width, value->numel() / value->dims()[0],
                      "output tensor should have the same shape with table "
                      "except the dims[0].");
    for (size_t i = 0; i < ids.numel(); ++i) {
      int64_t index = AutoGrownIndex(ids.data<int64_t>()[i], auto_grown);
      framework::VisitDataType(
          framework::ToDataType(value_->type()),
          TensorCopyVisitor(value, i * value_width, *value_.get(),
                            index * value_width, value_width));
    }
  }
}

}  // namespace framework
}  // namespace paddle
