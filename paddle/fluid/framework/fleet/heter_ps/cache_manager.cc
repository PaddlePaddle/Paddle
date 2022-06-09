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
#include <set>
#include <thread>
#include "glog/logging.h"
#include "paddle/fluid/framework/fleet/heter_ps/cache_manager.h"

#if defined(PADDLE_WITH_XPU_KP)

namespace paddle {
namespace framework {

CacheManager::CacheManager(): thread_num_(-1), batch_sz_(-1), worker_num_(1) {
#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  current_batch_fid_seq_lock = std::make_shared<std::mutex>();
#endif
}


void CacheManager::init(int thread_num, int batch_sz, int worker_num) {
  thread_num_ = thread_num;
  batch_sz_ = batch_sz;
  worker_num_ = worker_num;
  clear_sign2fids();
  VLOG(0) << "CacheManager init:" << thread_num << "|" << batch_sz << "|" << worker_num;
}

void CacheManager::clear_sign2fids() {
  sign2fid_.clear();
  fid2meta_.clear();
  feasign_cnt_ = 0;
}

void CacheManager::build_sign2fids(const FeatureKey* d_keys, size_t len) {
  VLOG(0) << "build_sign2fids: keylen:" << len;
  // pre-build the sign2fid_, in order not to use mutex
  for (size_t i = 0; i < len; ++i) {
    CHECK(sign2fid_.find(d_keys[i]) == sign2fid_.end()) 
        << "build_sign2fids: error, the same key found:" << d_keys[i];
    sign2fid_[d_keys[i]] = 0;
  }
  size_t origin_size = fid2meta_.size();
  fid2meta_.resize(origin_size + len);
  VLOG(0) << "build_sign2fids: resize fid2meta from " << origin_size << " to " << fid2meta_.size();
  // build sign 2 fids
  std::vector<std::thread> threads(thread_num_);
  size_t split_len = len % thread_num_ == 0 ? (len / thread_num_) : (len / thread_num_ + 1);
  for (int i = 0; i < thread_num_; ++i) {
    threads[i] = std::thread([i, this](const FeatureKey* keys, size_t keys_len) {
      for (size_t j = 0; j < keys_len; ++j) {
          int tmp = feasign_cnt_++;
          sign2fid_[keys[j]] = tmp;
          fid2meta_[tmp] = {keys[j]};
      }
    }, &d_keys[i * split_len], std::min(split_len, len - i * split_len));
  }
  for (auto & thd : threads) {
    thd.join();
  }
  VLOG(0) << "build_sign2fids: exit";
}

uint64_t CacheManager::query_sign2fid(const FeatureKey & key) {
  //VLOG(0) << "query_sign2fid:" << key << "->" << sign2fid_[key];
  return sign2fid_[key];
}

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

void CacheManager::build_batch_fid_seq(std::deque<Record> & recs) {
  int size = recs.size();
  // init prebuild n_batch_bfidseq vector
  int n_batch_sz = batch_sz_ * worker_num_;
  int groups = size % n_batch_sz == 0 ? (size / n_batch_sz) : (size / n_batch_sz) + 1;
  std::vector<std::shared_ptr<std::vector<uint64_t>>> n_batch_bfidseq(groups, nullptr);
  VLOG(0) << "build_batch_fid_seq: in size:" << size << "groups: " << groups;
  // fill n_batch_bfidseq vector by multi-thread
  std::vector<std::thread> threads(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    threads[i] = std::thread([this, i, size, &recs, n_batch_sz, &n_batch_bfidseq]() {
      VLOG(0) << "build_batch_fid_seq: in thread-" << i; 
      int my_group = 0;
      for (int batch_first = i * n_batch_sz; batch_first < size; batch_first += thread_num_ * n_batch_sz) {
        int current_batch_sz = std::min(n_batch_sz, size - batch_first);
        std::shared_ptr<std::vector<uint64_t>> current_bfid_seq = std::make_shared<std::vector<uint64_t>>();
        std::set<uint64_t> current_bfid_set;
        auto it = recs.begin() + batch_first;
        for (int j = 0; j < current_batch_sz; ++j) {
            const Record & cur_rec = *(it + j);
            for (auto & fea : cur_rec.uint64_feasigns_) {
              current_bfid_set.insert(fea.sign().uint64_feasign_); // feasign already converted to fid
            }
        }
        current_bfid_seq->assign(current_bfid_set.begin(), current_bfid_set.end());
        // fid_seq_channel_->Put(current_bfid_seq);
        //for (auto it = current_bfid_set.begin(); it != current_bfid_set.end(); it++) {
        //    VLOG(0) << "bfid_set item:" << *it;
        //}
        n_batch_bfidseq[my_group * thread_num_ + i] = current_bfid_seq;
        ++my_group;
        // VLOG(0) << "build_batch_fid_seq:" 
        //         << recs.size() << "|"
        //         << fid_seq_channel_->Size() << "|"
        //         << batch_first << "|" 
        //         << current_batch_sz << "|" 
        //         << current_bfid_seq->size();
        }
    });
  }
  for (auto & thd : threads) {
    thd.join();
  }
  // check for debug
  for (auto group_bfid_ptr : n_batch_bfidseq) {
    if (group_bfid_ptr == nullptr) {
      VLOG(0) << "err: bfid_ptr is nullptr"; 
    } else {
      VLOG(0) << "group size: " << group_bfid_ptr->size();
    }
  }
  // write n_batch_bfidseq to channel
  fid_seq_channel_->Write(groups, &n_batch_bfidseq[0]);
  fid_seq_channel_->Close();
}

void CacheManager::prepare_current_batch_fid_seq() {
  std::lock_guard<std::mutex> lock(*current_batch_fid_seq_lock);
  if (current_batch_fid_seq_ == nullptr || current_batch_fid_seq_ref_ == worker_num_) {
    current_batch_fid_seq_ref_ = 0;
    if (!fid_seq_channel_->Get(current_batch_fid_seq_)) {
      current_batch_fid_seq_ = nullptr;
    } else {
      for (uint64_t i = 0; i < current_batch_fid_seq_->size(); ++i) {
          current_batch_fid2bfid_[(*current_batch_fid_seq_)[i]] = i;
      }
    }
  } else {
    current_batch_fid_seq_ref_++;
  }
}

std::shared_ptr<std::vector<uint64_t>>  CacheManager::get_current_batch_fid_seq() {
  CHECK(current_batch_fid_seq_ != nullptr);
  return current_batch_fid_seq_; 
}

void CacheManager::convert_fid2bfid(const uint64_t * fids, int * out_bfids, int size) {
  std::lock_guard<std::mutex> lock(*current_batch_fid_seq_lock);
  for (int i = 0; i < size; ++i) {
    out_bfids[i] = current_batch_fid2bfid_[fids[i]];
  }
}

#endif
}  // end namespace framework
}  // end namespace paddle

#endif
