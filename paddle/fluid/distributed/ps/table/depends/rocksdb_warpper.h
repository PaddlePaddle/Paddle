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

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <iostream>
#include <string>

namespace paddle {
namespace distributed {

class RocksDBHandler {
 public:
  RocksDBHandler() {}
  ~RocksDBHandler() {}

  static RocksDBHandler* GetInstance() {
    static RocksDBHandler handler;
    return &handler;
  }

  int initialize(const std::string& db_path, const int colnum) {
    VLOG(3) << "db path: " << db_path << " colnum: " << colnum;
    rocksdb::Options options;
    rocksdb::BlockBasedTableOptions bbto;
    bbto.block_size = 4 * 1024;
    bbto.block_cache = rocksdb::NewLRUCache(64 * 1024 * 1024);
    bbto.block_cache_compressed = rocksdb::NewLRUCache(64 * 1024 * 1024);
    bbto.cache_index_and_filter_blocks = false;
    bbto.filter_policy.reset(rocksdb::NewBloomFilterPolicy(20, false));
    bbto.whole_key_filtering = true;
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(bbto));

    options.keep_log_file_num = 100;
    options.max_log_file_size = 50 * 1024 * 1024;  // 50MB
    options.create_if_missing = true;
    options.use_direct_reads = true;
    options.max_background_flushes = 5;
    options.max_background_compactions = 5;
    options.base_background_compactions = 10;
    options.write_buffer_size = 256 * 1024 * 1024;  // 256MB
    options.max_write_buffer_number = 8;
    options.max_bytes_for_level_base =
        options.max_write_buffer_number * options.write_buffer_size;
    options.min_write_buffer_number_to_merge = 1;
    options.target_file_size_base = 1024 * 1024 * 1024;  // 1024MB
    options.memtable_prefix_bloom_size_ratio = 0.02;
    options.num_levels = 4;
    options.max_open_files = -1;

    options.compression = rocksdb::kNoCompression;
    options.level0_file_num_compaction_trigger = 8;
    options.level0_slowdown_writes_trigger =
        1.8 * options.level0_file_num_compaction_trigger;
    options.level0_stop_writes_trigger =
        3.6 * options.level0_file_num_compaction_trigger;

    if (!db_path.empty()) {
      std::string rm_cmd = "rm -rf " + db_path;
      system(rm_cmd.c_str());
    }

    rocksdb::Status s = rocksdb::DB::Open(options, db_path, &_db);
    assert(s.ok());
    _handles.resize(colnum);
    for (int i = 0; i < colnum; i++) {
      s = _db->CreateColumnFamily(options, "shard_" + std::to_string(i),
                                  &_handles[i]);
      assert(s.ok());
    }
    LOG(INFO) << "DB initialize success, colnum:" << colnum;
    return 0;
  }

  int put(int id, const char* key, int key_len, const char* value,
          int value_len) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::Status s =
        _db->Put(options, _handles[id], rocksdb::Slice(key, key_len),
                 rocksdb::Slice(value, value_len));
    assert(s.ok());
    return 0;
  }

  int put_batch(int id, std::vector<std::pair<char*, int>>& ssd_keys,
                std::vector<std::pair<char*, int>>& ssd_values, int n) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::WriteBatch batch(n * 128);
    for (int i = 0; i < n; i++) {
      batch.Put(_handles[id],
                rocksdb::Slice(ssd_keys[i].first, ssd_keys[i].second),
                rocksdb::Slice(ssd_values[i].first, ssd_values[i].second));
    }
    rocksdb::Status s = _db->Write(options, &batch);
    assert(s.ok());
    return 0;
  }

  int get(int id, const char* key, int key_len, std::string& value) {
    rocksdb::Status s = _db->Get(rocksdb::ReadOptions(), _handles[id],
                                 rocksdb::Slice(key, key_len), &value);
    if (s.IsNotFound()) {
      return 1;
    }
    assert(s.ok());
    return 0;
  }

  int del_data(int id, const char* key, int key_len) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::Status s =
        _db->Delete(options, _handles[id], rocksdb::Slice(key, key_len));
    assert(s.ok());
    return 0;
  }

  int flush(int id) {
    rocksdb::Status s = _db->Flush(rocksdb::FlushOptions(), _handles[id]);
    assert(s.ok());
    return 0;
  }

  rocksdb::Iterator* get_iterator(int id) {
    return _db->NewIterator(rocksdb::ReadOptions(), _handles[id]);
  }

  int get_estimate_key_num(uint64_t& num_keys) {
    _db->GetAggregatedIntProperty("rocksdb.estimate-num-keys", &num_keys);
    return 0;
  }

 private:
  std::vector<rocksdb::ColumnFamilyHandle*> _handles;
  rocksdb::DB* _db;
};
}  // distributed
}  // paddle
