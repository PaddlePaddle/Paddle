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

#include "paddle/fluid/framework/fleet/heter_ps/cache_manager.h"

#if defined(PADDLE_WITH_XPU_KP)

namespace paddle {
namespace framework {

CacheManager::CacheManager(int worker_num): thread_num_(-1), batch_sz_(-1), worker_num_(worker_num) {
#if defined(PADDLE_WITH_XPU_CACHE_BFID)
    current_batch_fid_seq_lock = std::make_shared<std::mutex>();
#endif
}

CacheManager::CacheManager(int thread_num, int batch_sz, int worker_num): 
        thread_num_(thread_num), batch_sz_(batch_sz), worker_num_(worker_num) {
#if defined(PADDLE_WITH_XPU_CACHE_BFID)
    current_batch_fid_seq_lock = std::make_shared<std::mutex>();
#endif
}

void CacheManager::build_sign2fids(FeatureKey* d_keys, size_t len) {
    sign2fid_.clear();
    // pre-build the sign2fid_, in order not to use mutex
    for (size_t i = 0; i < len; ++i) {
        sign2fid_[d_keys[i]] = 0;
    }
    fid2meta_ = std::vector<CacheMeta>(len);

    // build sign 2 fids
    std::vector<std::thread> threads(thread_num_);
    size_t split_len = len % thread_num_ == 0 ? (len / thread_num_) : (len / thread_num_ + 1);
    feasign_cnt_ = 0;
    for (int i = 0; i < thread_num_; ++i) {
        threads[i] = std::thread([this](FeatureKey* keys, size_t keys_len) {
            for (size_t j = 0; j < keys_len; ++j) {
                int tmp = feasign_cnt_;
                while (!feasign_cnt_.compare_exchange_strong(tmp, tmp + 1)) {}
                sign2fid_[keys[j]] = tmp;
                fid2meta_[tmp] = {keys[j]};
            }
        }, &d_keys[i * split_len], std::min(split_len, len - i * split_len));
    }
    for (auto & thd : threads) {
        thd.join();
    }
}

uint64_t CacheManager::query_sign2fid(FeatureKey & key) {
    return sign2fid_[key];
}

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

void CacheManager::build_batch_fid_seq(Record * recs, int size) {
    std::vector<std::thread> threads(thread_num_);
    for (int i = 0; i < thread_num_; ++i) {
        threads[i] = std::thread([this, i, recs, size]() {
            for (int batch_first = i * batch_sz_; batch_first < size; batch_first += thread_num_ * batch_sz_) {
                int current_batch_sz = std::min(batch_sz_, size - batch_first);
                std::shared_ptr<std::vector<uint64_t>> current_bfid_seq = std::make_shared<std::vector<uint64_t>>();
                std::set<uint64_t> current_bfid_set;
                for (int j = 0; j < current_batch_sz; ++j) {
                    const Record & cur_rec = recs[batch_first + j];
                    for (auto & fea : cur_rec.uint64_feasigns_) {
                        current_bfid_set.insert(sign2fid_[fea.sign().uint64_feasign_]);
                    }
                }
                current_bfid_seq->assign(current_bfid_set.begin(), current_bfid_set.end());
                fid_seq_channel_->Put(current_bfid_seq);
            }
        });
    }
    for (auto & thd : threads) {
        thd.join();
    }
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
    return current_batch_fid_seq_; 
}

void CacheManager::convert_fid2bfid(uint64_t * fids, int * out_bfids, int size) {
    std::lock_guard<std::mutex> lock(*current_batch_fid_seq_lock);
    for (int i = 0; i < size; ++i) {
        out_bfids[i] = current_batch_fid2bfid_[fids[i]];
    }
}

#endif

}  // end namespace framework
}  // end namespace paddle

#endif
