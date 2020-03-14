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

DEFINE_int32(dist_lookuptable_shard_num, 13,
             "the number of shard on this pserver");

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
  void apply() const {
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

struct TensorFillVisitor {
  TensorFillVisitor(framework::Tensor* dst, int64_t dst_offset, int64_t size,
                    float value)
      : dst_(dst), dst_offset_(dst_offset), size_(size) {}

  template <typename T>
  void apply() const {
    // TODO(qiao): support other place
    platform::CPUPlace cpu;
    auto* tensor_data = dst_->mutable_data<T>(cpu);
    auto* start = tensor_data + dst_offset_;
    auto* end = start + size_;
    std::fill(start, end, static_cast<T>(0.0));
  }

  framework::Tensor* dst_;
  int64_t dst_offset_;
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

SelectedRows::SelectedRows(const std::vector<int64_t>& rows,
                           const int64_t& height)
    : rows_(rows),
      height_(height),
      shard_num_(FLAGS_dist_lookuptable_shard_num) {
  VLOG(3) << "shard_num_ " << shard_num_;
  value_.reset(new Tensor());
}

SelectedRows::SelectedRows() : shard_num_(FLAGS_dist_lookuptable_shard_num) {
  VLOG(3) << "shard_num_ " << shard_num_;
  height_ = 0;
  value_.reset(new Tensor());
}


bool SelectedRows::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

void SelectedRows::InitDataShards() {
  if (data_shards_.size() == shard_num_) {
    return;
  }
  PADDLE_ENFORCE_GT(value_->numel(), 0,
                    "tensor should be inited when call InitDataShards");
  int64_t shard_size = value_->dims()[0] / shard_num_;
  PADDLE_ENFORCE_GT(shard_size, 0, "shard_size should be larger then 0");
  VLOG(3) << "InitDataShards, shard_num_=" << shard_num_
          << " shard_size=" << shard_size;
  for (int64_t i = 0; i < shard_num_; ++i) {
    data_shards_.emplace_back(new DataShard(i, shard_size));
  }
}

int64_t SelectedRows::GetIndexById(int64_t id,
                                bool auto_grown,
                                bool is_test){
  size_t shard_id = ShardId(id);
  return data_shards_[shard_id]->GetIndexById(id, auto_grown, is_test);
}


void SelectedRows::GetIndexsByIds(const std::vector<int64_t>& ids,
                                  std::vector<int64_t>* indexs,
                                  bool auto_grown,
                                  bool is_test) {
  PADDLE_ENFORCE_EQ(data_shards_.size(), shard_num_,
                    "data shards is not inited");
  std::vector<std::vector<std::pair<int64_t, int64_t>>> re_sharded_keys(
      shard_num_);
  for (size_t i = 0; i < ids.size(); ++i) {
    auto id = ids[i];
    size_t shard_id = ShardId(id);
    re_sharded_keys[shard_id].emplace_back(std::make_pair(id, i));
  }
  std::vector<std::future<void>> fs;
  std::vector<std::future<void>> futures(shard_num_);
  for (size_t i = 0; i < shard_num_; ++i) {
    if (re_sharded_keys[i].size() == 0) {
      continue;
    }
    fs.push_back(
      framework::Async([i, this, &re_sharded_keys, &indexs,
                        auto_grown, is_test]() {
        this->data_shards_[i]->GetIndexsByIds(re_sharded_keys[i], indexs, auto_grown, is_test);
    }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

void SelectedRows::SyncIndex() {
  VLOG(1) << "SyncBeforeSave";
  // id_to_index_.clear();
  rows_.clear();
  rows_.reserve(id_to_index_.size() * 2);
  
  std::vector<int64_t> row_ids;
  std::vector<int64_t> row_indexs;
  row_ids.reserve(id_to_index_.size());
  row_indexs.reserve(id_to_index_.size());
  for (auto& shard : data_shards_) {
    shard->GetAllIdToAbsOffset(row_ids, row_indexs);
  }
  rows_.insert(rows_.end(), row_ids.begin(), row_ids.end());
  rows_.insert(rows_.end(), row_indexs.begin(), row_indexs.end());

}

void SelectedRows::ReconstructShardAfterLoad() {
  VLOG(0) << "SyncAfterLoad";
  InitDataShards();
  std::vector<std::unordered_map<int64_t, int64_t>> shard_id_to_offset;
  shard_id_to_offset.resize(shard_num_);
  size_t rows_size = rows_.size();
  PADDLE_ENFORCE_EQ(rows_size % 2, 0, "rows should have n * 2 elements");
  for (size_t i = 0; i < rows_size / 2; ++i) {
    int64_t id = rows_[i];
    int64_t abs_offset = rows_[rows_size / 2 + i];
    shard_id_to_offset[ShardId(id)][id] = abs_offset;
  }
  for (size_t shard_id = 0; shard_id < shard_id_to_offset.size(); ++shard_id) {
    data_shards_[shard_id]->ReconstructShardIndex(shard_id_to_offset[shard_id]);
  }
}

void SelectedRows::Get(const framework::Tensor& ids, framework::Tensor* value,
                       bool auto_grown, bool is_test) {
  PADDLE_ENFORCE(value->IsInitialized(),
                 "The value tensor should be initialized.");
  if (ids.numel() == 0) {
    VLOG(3) << "keys is empty, please check data!";
  } else {
    int64_t value_width = value_->numel() / value_->dims()[0];
    PADDLE_ENFORCE_EQ(value_width, value->numel() / value->dims()[0],
                      "output tensor should have the same shape with table "
                      "except the dims[0].");
    std::vector<int64_t> all_ids(ids.numel());
    auto* ids_data = ids.data<int64_t>();
    const size_t ids_num = ids.numel();
    for (auto i = 0; i < ids_num; ++i) {
      all_ids[i] = ids_data[i];
    }

    std::vector<int64_t> id_indexes(ids.numel());
    GetIndexsByIds(all_ids, &id_indexes, auto_grown, is_test);
    int64_t table_height = value_->dims()[0];
    for (int i = 0; i < ids_num; ++i) {
      auto id = ids.data<int64_t>()[i];
      int64_t index = id_indexes[i];
      PADDLE_ENFORCE_LT(index, table_height,
                        "index should be less then table height");
      if (index < 0) {
        VLOG(5) << "id " << id << " not in the table, return 0";
        framework::VisitDataType(
            value_->type(),
            TensorFillVisitor(value, i * value_width, value_width, 0.0));
      } else {
        framework::VisitDataType(
            value_->type(),
            TensorCopyVisitor(value, i * value_width, *value_.get(),
                              index * value_width, value_width));
      }
    }
  }
}

}  // namespace framework
}  // namespace paddle
