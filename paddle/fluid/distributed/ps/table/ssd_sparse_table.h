// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/distributed/ps/table/common_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/depends/rocksdb_warpper.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace distributed {
class SSDSparseTable : public CommonSparseTable {
 public:
  SSDSparseTable() {}
  virtual ~SSDSparseTable() {}

  virtual int32_t Initialize() override;

  void SaveMetaToText(std::ostream* os, const CommonAccessorParameter& common,
                      const size_t shard_idx, const int64_t total);

  int64_t SaveValueToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
                          std::shared_ptr<::ThreadPool> pool, const int mode,
                          int shard_id);

  virtual int64_t LoadFromText(
      const std::string& valuepath, const std::string& metapath,
      const int pserver_id, const int pserver_num, const int local_shard_num,
      std::vector<std::shared_ptr<ValueBlock>>* blocks);

  virtual int32_t Load(const std::string& path, const std::string& param);

  // exchange data
  virtual int32_t UpdateTable();

  virtual int32_t Pull(TableContext& context);
  virtual int32_t Push(TableContext& context);

  virtual int32_t PullSparse(float* values, const PullSparseValue& pull_value);

  virtual int32_t PullSparsePtr(char** pull_values, const uint64_t* keys,
                                size_t num);

  virtual int32_t Flush() override { return 0; }
  virtual int32_t Shrink(const std::string& param) override;
  virtual void Clear() override {}

 private:
  RocksDBHandler* _db;
  int64_t _cache_tk_size;
};

}  // namespace ps
}  // namespace paddle
#endif
