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
#include <fstream>
#include "glog/logging.h"
#include "paddle/fluid/framework/fleet/heter_ps/cache_manager.h"
#include "paddle/fluid/framework/fleet/heter_ps/log_patch.h"

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
  dev_feasign_cnts_.resize(worker_num, nullptr);
  for (int i = 0; i < worker_num; i++) {
    dev_feasign_cnts_[i] = std::make_shared<std::atomic<int>>(0);
  }
  VLOG(0) << "CacheManager init:" << thread_num << "|" << batch_sz << "|" << worker_num;
}

void CacheManager::clear_sign2fids() {
  sign2fid_.clear();
  fid2meta_.clear();
  dev_feasign_cnts_.resize(0);
#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  current_batch_fid2bfid_.clear();
  current_batch_fid_seq_ref_ = 0;
#endif
}

void CacheManager::build_sign2fids(const FeatureKey* d_keys, size_t len) {
  VLOG(0) << "build_sign2fids: keylen:" << len;
  // pre-build the sign2fid_, in order not to use mutex
  PADDLE_ENFORCE_GT(dev_feasign_cnts_.size(), 0,  
      platform::errors::External("maybe not call CacheManager::init()"));
  if (sign2fid_.find(0) == sign2fid_.end()) {
    // padding feasign 0, set fid 0, to be processed specially later in pull/push
    sign2fid_[0] = (*(dev_feasign_cnts_[0]))++;
    fid2meta_.resize(1);
    fid2meta_[0] = {0};
  }
  std::vector<int> fea_cnts;
  fea_cnts.resize(worker_num_, 0);
  for (size_t i = 0; i < len; ++i) {
    if (d_keys[i] == 0) {
      continue;
    }
    PADDLE_ENFORCE(sign2fid_.find(d_keys[i]) == sign2fid_.end(),
        platform::errors::External("the same key found:%llu", (unsigned long long)d_keys[i]));
    sign2fid_[d_keys[i]] = 0;
    fea_cnts[d_keys[i] % worker_num_]++;
  }
  int need_resize = 0;
  int origin_size = fid2meta_.size();
  int curr_reserve_size = int(origin_size / worker_num_); 
  for (int i = 0; i < worker_num_; i++) {
    int inc_cnt = fea_cnts[i];
    int curr_cnt = *(dev_feasign_cnts_[i]);
    if (curr_reserve_size - curr_cnt - inc_cnt < 0) {
      need_resize = (need_resize > (curr_cnt + inc_cnt)) ? need_resize : (curr_cnt + inc_cnt);
    }
  }
  if (need_resize > 0) {
    fid2meta_.resize(need_resize * worker_num_); 
  }
  VLOG(0) << "build_sign2fids: worker_num:" << worker_num_
          << ", curr_reserve_size:" << curr_reserve_size
          << ", need_resize:" << need_resize;
  VLOG(0) << "build_sign2fids: resize fid2meta from " << origin_size << " to " << fid2meta_.size();
  // build sign 2 fids
  std::vector<std::thread> threads(thread_num_);
  size_t split_len = len % thread_num_ == 0 ? (len / thread_num_) : (len / thread_num_ + 1);
  for (int i = 0; i < thread_num_; ++i) {
    threads[i] = std::thread([i, this](const FeatureKey* keys, size_t keys_len) {
      for (size_t j = 0; j < keys_len; ++j) {
        if (keys[j] == 0) {
          continue;
        }
        int dev_id = keys[j] % worker_num_;
        int tmp = (*(dev_feasign_cnts_[dev_id]))++;
        int tmp_fid = dev_id + tmp * worker_num_;  
        sign2fid_[keys[j]] = tmp_fid;
        PADDLE_ENFORCE_LT(tmp_fid, (int)fid2meta_.size(),
            platform::errors::External("fid2meta_ size too small"));
        fid2meta_[tmp_fid] = {keys[j]};
      }
    }, &d_keys[i * split_len], std::min(split_len, len - i * split_len));
  }
  for (auto & thd : threads) {
    thd.join();
  }
  VLOG(0) << "build_sign2fids: exit";
}

uint32_t CacheManager::query_sign2fid(const FeatureKey & key) {
  //VLOG(0) << "query_sign2fid:" << key << "->" << sign2fid_[key];
  return sign2fid_[key];
}

uint32_t CacheManager::get_max_fid() {
  uint32_t max_id = 0;
  for (int i = 0; i < worker_num_; i++) {
    uint32_t curr_cnt = *(dev_feasign_cnts_[i]);
    if (curr_cnt == 0) {
      continue;
    }
    uint32_t fid = i + (curr_cnt - 1) * worker_num_;
    if (fid > max_id) {
      max_id = fid;
    }
  }
  return max_id;
}

void CacheManager::dump_to_file() {
  sleep(1);
  auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm* ptm = localtime(&now_time);
  char date[100] = {0};
  snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
      (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
      (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);

  std::stringstream name_ss;
  name_ss << "cache_manager." << date << ".dump";

  std::ofstream ofs;
  ofs.open(name_ss.str(), std::ios::app);
  for (size_t i = 0; i < fid2meta_.size(); i++) {
    int dev_id = i % worker_num_;
    int offset = int(i / worker_num_);
    if (offset >= *(dev_feasign_cnts_[dev_id])) {
      ofs << "fid2meta:" << i << " EMPTY EMPTY" << std::endl;
      continue;
    }
    ofs << "fid2meta:" << i << " " << fid2meta_[i].sign_;
    if (sign2fid_.find(fid2meta_[i].sign_) != sign2fid_.end()) {
      ofs << " " << sign2fid_[fid2meta_[i].sign_] << std::endl;
    } else {
      ofs << " " << "error-NULL" << std::endl;
    }
  }

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  if (fid_seq_channel_ != nullptr) {
    auto & chan_dq = fid_seq_channel_->GetData();
    auto iter = chan_dq.begin();
    
    for (size_t i = 0; i < chan_dq.size(); i++) {
      auto & fid_vec = *(*(iter + i));
      ofs << "fidseq:" << i << ">>>>>>>>>>>>>>>>>>>>>" << std::endl;
      for (size_t j = 0; j < fid_vec.size(); j++) {
        ofs << "fidseq:" << i << ":bfid-" << j << ":" << fid_vec[j] << std::endl;
      }
      ofs << "fidseq:" << i << "<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    }
  }
#endif
  ofs << "------------------------------------------" << std::endl;
  ofs.close(); 
}

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

void CacheManager::build_batch_fid_seq(
    std::vector<std::deque<Record> *> & all_chan_recs, 
               const std::vector<bool> & slot_is_dense) {
  PADDLE_ENFORCE_GT(all_chan_recs.size(), 0,
      platform::errors::External("all_chan_recs size error"));
  size_t expected_chan_size = all_chan_recs[0]->size();
  for (auto & chan_recs : all_chan_recs) {
    PADDLE_ENFORCE_EQ(chan_recs->size(), expected_chan_size);
  }
  
  int size = expected_chan_size;
  int batch_sz = batch_sz_;
  int groups = size % batch_sz == 0 ? (size / batch_sz) : (size / batch_sz) + 1;
  std::vector<std::shared_ptr<std::vector<uint32_t>>> n_batch_bfidseq(groups, nullptr);
  VLOG(0) << "build_batch_fid_seq: in size:" << size 
          << ", batch_size: " << batch_sz
          << ", groups: " << groups
          << ", channels:" << all_chan_recs.size();
  
  std::vector<std::thread> threads(thread_num_);
  for (int i = 0; i < thread_num_; ++i) { 
    threads[i] = std::thread([&, i, this]() {
      VLOG(0) << "build_batch_fid_seq: in thread-" << i;
      int my_group = 0;
      for (int batch_first = i * batch_sz; batch_first < size; batch_first += thread_num_ * batch_sz) { 
        int current_batch_sz = std::min(batch_sz, size - batch_first);

        // process batch data for every chan_recs
        std::shared_ptr<std::vector<uint32_t>> current_bfid_seq = std::make_shared<std::vector<uint32_t>>();
        std::set<uint32_t> current_bfid_set;
        std::set<int> slot_has_val;
        for (auto & recs : all_chan_recs) {
          auto it = recs->begin() + batch_first;
          for (int j = 0; j < current_batch_sz; ++j) {
            const Record & cur_rec = *(it + j);
            for (auto & fea : cur_rec.uint64_feasigns_) {
              slot_has_val.insert(fea.slot());
              PADDLE_ENFORCE_LT(fea.slot(), slot_is_dense.size());
              if (slot_is_dense[fea.slot()]) {
                continue;
              }
              current_bfid_set.insert((uint32_t)fea.sign().uint64_feasign_); // feasign already converted to fid(uint32_t)
            }
          }
        } // process finished
        if (slot_has_val.size() < slot_is_dense.size()) {
          current_bfid_set.insert(0); // add 0 as padding feasign
        }
        current_bfid_seq->assign(current_bfid_set.begin(), current_bfid_set.end());
        n_batch_bfidseq[my_group * thread_num_ + i] = current_bfid_seq;
        ++my_group;
      }
    });
  }

  for (auto & thd : threads) {
    thd.join();
  }
  // check for debug
  for (auto group_bfid_ptr : n_batch_bfidseq) {
    PADDLE_ENFORCE_NOT_NULL(group_bfid_ptr, 
        platform::errors::NotFound("batch fid sequence has nullptr item"));
    VLOG(3) << "group size: " << group_bfid_ptr->size();
  }
  // write n_batch_bfidseq to channel
  fid_seq_channel_ = paddle::framework::MakeChannel<std::shared_ptr<std::vector<uint32_t>>>();
  fid_seq_channel_->Write(groups, &n_batch_bfidseq[0]);
  fid_seq_channel_->Close();
}

void CacheManager::prepare_current_batch_fid_seq() {
  std::lock_guard<std::mutex> lock(*current_batch_fid_seq_lock);
  if (current_batch_fid_seq_ == nullptr || current_batch_fid_seq_ref_ == worker_num_) {
    current_batch_fid_seq_ref_ = 0;
    current_batch_fid2bfid_.clear();
    if (!fid_seq_channel_->Get(current_batch_fid_seq_)) {
      current_batch_fid_seq_ = nullptr;
    } else {
      for (size_t i = 0; i < current_batch_fid_seq_->size(); ++i) {
        current_batch_fid2bfid_[(*current_batch_fid_seq_)[i]] = i;
      }
      current_batch_fid_seq_ref_++;
    }
  } else {
    current_batch_fid_seq_ref_++;
  }
}

std::shared_ptr<std::vector<uint32_t>>  CacheManager::get_current_batch_fid_seq() {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fid_seq_);
  return current_batch_fid_seq_; 
}

void CacheManager::convert_fid2bfid(const uint32_t * fids, int * out_bfids, int size) {
  std::lock_guard<std::mutex> lock(*current_batch_fid_seq_lock);
  for (int i = 0; i < size; ++i) {
    PADDLE_ENFORCE(current_batch_fid2bfid_.find(fids[i]) != current_batch_fid2bfid_.end(),
        platform::errors::External("fid not found:%ul", (unsigned long)fids[i])); 
    out_bfids[i] = current_batch_fid2bfid_[fids[i]];
  }
}

uint64_t CacheManager::convert_fid2feasign(const uint32_t fid) {
  return fid2meta_[fid].sign_;
}

void CacheManager::debug() {
  for (auto iter = current_batch_fid2bfid_.begin(); iter != current_batch_fid2bfid_.end(); iter++) {
    VLOG(0) << "cache manager debug:" << iter->first << " " << iter -> second;
  }
}

#endif
}  // end namespace framework
}  // end namespace paddle

#endif
