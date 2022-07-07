/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_XPU_KP)

class CacheManager {
 public:
  struct CacheMeta {
   public:
    uint64_t sign_;
  };

  CacheManager();
  ~CacheManager() {}

  // Todo: need check init state before every function call
  void init(int thread_num, int batch_sz, int worker_num);

  void clear_sign2fids();
  void build_sign2fids(const FeatureKey* d_keys, size_t len);
  uint32_t query_sign2fid(const FeatureKey & key);
  uint32_t get_max_fid();

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  void build_batch_fid_seq(
      std::vector<std::deque<Record> *> & all_chan_recs, 
                const std::vector<bool> & slot_is_dense);
  void prepare_current_batch_fid_seq();
  std::shared_ptr<std::vector<uint32_t>>  get_current_batch_fid_seq();
  void convert_fid2bfid(const uint32_t * fids, int * out_bfids, int size);
  uint64_t convert_fid2feasign(const uint32_t fids);
  void debug();
#endif

  void dump_to_file();

 private:
  std::vector<std::shared_ptr<std::atomic<int>>> dev_feasign_cnts_;
  std::unordered_map<FeatureKey, uint32_t> sign2fid_;
  std::vector<CacheMeta> fid2meta_;

  int thread_num_;
  int batch_sz_;
  int worker_num_;

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  // for batch fid sequence
  std::shared_ptr<paddle::framework::ChannelObject<
      std::shared_ptr<std::vector<uint32_t>>>> fid_seq_channel_ = nullptr;
  std::shared_ptr<std::vector<uint32_t>> current_batch_fid_seq_ = nullptr;
  std::unordered_map<uint32_t, int> current_batch_fid2bfid_;
  int current_batch_fid_seq_ref_ = 0;
  std::shared_ptr<std::mutex> current_batch_fid_seq_lock;
#endif

};

#endif
}  // end namespace framework
}  // end namespace paddle
