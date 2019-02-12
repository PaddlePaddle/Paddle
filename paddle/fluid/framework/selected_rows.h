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

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include <ThreadPool.h>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace framework {

/*
 * split selected rows into multiple shards, each shard will has an thread to
 * update it.
 */
class DataShard {
 public:
  DataShard(int64_t shard_id, int64_t shard_size)
      : shard_id_(shard_id),
        shard_size_(shard_size),
        pool_(new ::ThreadPool(1)) {}

  std::future<void> GetIndexsByIds(
      const std::vector<std::pair<int64_t, int64_t>> id_to_offsets,
      std::vector<int64_t>* value_indexs, bool auto_grown) {
    auto task = [this, &id_to_offsets, value_indexs, auto_grown] {
      for (auto& id_to_offset : id_to_offsets) {
        int64_t id = id_to_offset.first;
        int64_t offset = id_to_offset.second;
        (*value_indexs)[offset] = GetIndexById(id, auto_grown);
      }
    };
    return pool_->enqueue(std::move(task));
  }

 private:
  // this method did not support multithread!
  int64_t GetIndexById(int64_t id, bool auto_grown) {
    auto iter = id_to_offset_.find(id);
    if (iter == id_to_offset_.end()) {
      if (auto_grown) {
        auto offset = id_to_offset_.size();
        id_to_offset_[id] = offset;
        return shard_id_ * shard_size_ + offset;
      } else {
        return -1;
      }
    } else {
      return shard_id_ * shard_size_ + iter->second;
    }
  }

  std::unordered_map<int64_t, int64_t> id_to_offset_;
  int64_t shard_id_;
  int64_t shard_size_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
};

class SelectedRows {
  /*
   * @brief We can use the SelectedRows structure to reproduce a sparse table.
   *  A sparse table is a key-value structure that the key is an `int64_t`,
   *  and the value is a Tensor which the first dimension is 0.
   *  You can use the following interface to operate the sparse table, and you
   * can find
   *  some detail information from the comments of each interface:
   *
   *  HasKey(key), whether the sparse table has the specified key.
   *  Set(key, value), set a key-value pair into the sparse table.
   *  Get(keys, value*), get value by given key list and apply it to the given
   * value pointer
   *    with the specified offset.
   *
   */
 public:
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
    rwlock_.reset(new RWLock);
  }

  SelectedRows() {
    height_ = 0;
    value_.reset(new Tensor());
    rwlock_.reset(new RWLock);
  }

  platform::Place place() const { return value_->place(); }

  const Tensor& value() const { return *value_; }

  Tensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const Vector<int64_t>& rows() const { return rows_; }

  Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const Vector<int64_t>& rows) { rows_ = rows; }

  void InitDataShards() {
    PADDLE_ENFORCE(value_->IsInitialized(),
                   "tensor should be inited when call InitDataShards");
    int64_t shard_size = value_->dims()[0] / shard_num_;
    for (int64_t i = 0; i < shard_num_; ++i) {
      data_shards_.emplace_back(new DataShard(i, shard_size));
    }
  }

  /*
   * @brief Get the index of key in rows
   *
   * @return -1 if the key does not exists.
   */
  int64_t Index(int64_t key) const {
    auto it = std::find(rows_.begin(), rows_.end(), key);
    if (it == rows_.end()) {
      PADDLE_THROW("id %s not in table", key);
    }
    return static_cast<int64_t>(std::distance(rows_.begin(), it));
  }

  /*
   * @brief whether has the specified key in the table.
   *
   * @return true if the key is exists.
   */
  bool HasKey(int64_t key) const;

  /*
   * @brief Get value by the key list.
   * Note!!! this interface is only used when selected_rows is used as
   * parameters
   * for distribute lookup table.
   *
   * @return a list of pair which contains the non-exists key and the index in
   * the value
   */
  void Get(const framework::Tensor& ids, framework::Tensor* value,
           bool auto_grown = false, bool is_test = false);

  /*
   * @brief Get the index of the key from id_to_index_ map. If the key not
   * exist,
   * add the key into id_to_index_.
   *
   * Note!!! this interface is only used when selected_rows is used as
   * parameters
   * for distribute lookup table.
   *
   * @return index of the key.
   */
  int64_t AutoGrownIndex(int64_t key, bool auto_grown, bool is_test = false);

  void GetIndexsByIds(const std::vector<int64_t>& ids,
                      std::vector<int64_t>* indexs, bool auto_grown) {
    std::vector<std::vector<std::pair<int64_t, int64_t>>> re_sharded_keys(
        shard_num_);
    for (size_t i = 0; i < ids.size(); ++i) {
      auto id = ids[i];
      size_t shard_id = id % shard_num_;
      re_sharded_keys[shard_id].push_back(std::make_pair(id, i));
    }

    std::vector<std::future<void>> futures(shard_num_);
    for (size_t i = 0; i < shard_num_; ++i) {
      futures[i] = data_shards_[i]->GetIndexsByIds(re_sharded_keys[i], indexs,
                                                   auto_grown);
    }
    for (size_t i = 0; i < shard_num_; ++i) {
      futures[i].wait();
    }
  }

  /*
   * @brief Get the index of the key from id_to_index_ map.
   */
  inline int64_t GetIndexFromId(int64_t key) {
    auto iter = id_to_index_.find(key);
    if (iter == id_to_index_.end()) {
      return -1;
    } else {
      return iter->second;
    }
  }

  void SyncIndex();
  /*
   * @brief Get complete Dims before
   */
  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRows are simply concated when adding together. Until a
  // SelectedRows add a Tensor, will the duplicate rows be handled.
  Vector<int64_t> rows_;
  std::unordered_map<int64_t, int64_t>
      id_to_index_;  // should not be used when rows_ has duplicate member
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;  // height indicates the underline tensor's height
  std::unique_ptr<RWLock> rwlock_{nullptr};
  std::vector<std::unique_ptr<DataShard>> data_shards_;
  const int64_t shard_num_ = 13;  // magic number
};

/*
 * Serialize/Desiralize SelectedRows to std::ostream
 * You can pass ofstream or ostringstream to serilize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx);

}  // namespace framework
}  // namespace paddle
