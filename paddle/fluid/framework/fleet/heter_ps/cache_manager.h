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

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_XPU_KP)

class CacheManager {
 public:
  struct CacheMeta {
   public:
    uint64_t sign_;
  };

  CacheManager() {}
  ~CacheManager() {}

  void build_sign2fids(FeatureKey* d_keys, size_t len) {}
  uint64_t query_sign2fid(FeatureKey & key) { return 0; }

  void build_batch_fid_seq(Record * recs, int size) {}
  void prepare_current_batch_fid_seq() {}
  std::shared_ptr<std::vector<uint64_t>>  get_current_batch_fid_seq() { return nullptr; }
  void convert_fid2bfid(uint64_t * fids, uint64_t * out_bfids, int size) {}

 private:
  std::unordered_map<FeatureKey, uint64_t> sign2fid_;
  std::vector<CacheMeta> fid2meta_;

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  // for batch fid sequence
  std::shared_ptr<paddle::framework::ChannelObject<
      std::shared_ptr<std::vector<uint64_t>>>> 
      fid_seq_channel_ = 
          paddle::framework::MakeChannel<std::shared_ptr<std::vector<uint64_t>>>();;
  std::shared_ptr<std::vector<uint64_t>> current_batch_fid_seq_ = nullptr;
  std::unordered_map<uint64_t, uint64_t> current_batch_fid2bfid;
  int current_batch_fid_seq_ref_ = 0;
  std::shared_ptr<std::mutex> current_batch_fid_seq_lock;
#endif

};

#endif
}  // end namespace framework
}  // end namespace paddle
