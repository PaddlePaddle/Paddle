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
#include <algorithm>
#include "glog/logging.h"
#include <gflags/gflags.h>
#include "paddle/fluid/framework/fleet/heter_ps/thread_barrier.h"
#include "paddle/fluid/framework/fleet/heter_ps/cache_manager.h"
#include "paddle/fluid/framework/fleet/heter_ps/log_patch.h"

#if defined(PADDLE_WITH_XPU_KP)

DECLARE_bool(dump_cache_manager);

namespace paddle {
namespace framework {

CacheManager::CacheManager(std::shared_ptr<HeterPsResource> resource): 
    resource_(resource), thread_num_(-1), batch_sz_(-1), worker_num_(1) {
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

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  worker_barrier_.reset(worker_num);
  comm_streams_.resize(worker_num);
  for (int i = 0; i < worker_num; i++) {
    int dev_id = resource_->dev_id(i);
    AnyDeviceGuard guard(dev_id);
    auto & stream = comm_streams_[i];
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&stream));
  }
  prepare_merge_grad_threads_.resize(worker_num);
#endif

  if (FLAGS_dump_cache_manager) {
    debug_data_chan_ = paddle::framework::MakeChannel<std::string>();
  }
}

void CacheManager::clear_sign2fids() {
  sign2fid_.clear();
  fid2meta_.clear();
  dev_feasign_cnts_.resize(0);
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
  VLOG(3) << "build_sign2fids: exit";
}

uint32_t CacheManager::query_sign2fid(const FeatureKey & key) {
  return sign2fid_[key];
}

uint64_t CacheManager::query_fid2sign(const uint32_t & fid) {
  return fid2meta_[fid].sign_;
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

std::string CacheManager::dump_to_file() {
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

  if (debug_data_chan_ != nullptr) {
    if (current_batch_fidseq_ != nullptr) {
      std::string tmp_data = current_batch_fidseq_->to_string();
      debug_data_chan_->Put(std::move(tmp_data));
    }

    std::string debug_data;
    debug_data_chan_->Close();
    VLOG(0) << "debug_data_chan size:" << debug_data_chan_->Size();
    while (debug_data_chan_->Get(debug_data)) {
      ofs << debug_data;
    }
  }

  ofs << "------------------------------------------" << std::endl;
  ofs.close();
  return name_ss.str(); 
}

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

std::shared_ptr<std::vector<std::shared_ptr<std::vector<uint32_t>>>>
  CacheManager::parse_all_fidseq(std::vector<std::deque<Record> *> & all_chan_recs,
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
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<uint32_t>>>> n_batch_bfidseq_ptr =
      std::make_shared<std::vector<std::shared_ptr<std::vector<uint32_t>>>>(groups, nullptr);
  auto & n_batch_bfidseq = *n_batch_bfidseq_ptr;
  VLOG(0) << "parse_all_fidseq: in size:" << size
          << ", batch_size: " << batch_sz
          << ", groups: " << groups
          << ", channels:" << all_chan_recs.size();

  std::vector<std::thread> threads(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    threads[i] = std::thread([&, i, this]() {
      VLOG(3) << "parse_all_fidseq: in thread-" << i;
      int my_group = 0;
      for (int batch_first = i * batch_sz; batch_first < size; batch_first += thread_num_ * batch_sz) {
        int current_batch_sz = std::min(batch_sz, size - batch_first);

        // process batch data for every chan_recs
        std::shared_ptr<std::vector<uint32_t>> current_bfidseq = std::make_shared<std::vector<uint32_t>>();
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
        current_bfidseq->assign(current_bfid_set.begin(), current_bfid_set.end());
        n_batch_bfidseq[my_group * thread_num_ + i] = current_bfidseq;
        ++my_group;
      }
    });
  }
  for (auto & thd : threads) {
    thd.join();
  }
  return n_batch_bfidseq_ptr;
}

void CacheManager::build_batch_fidseq(
    std::vector<std::deque<Record> *> & all_chan_recs, 
               const std::vector<bool> & slot_is_dense) {
  platform::Timer timeline;
  timeline.Start();

  if (batch_fidseq_proc_thread_.joinable()) {
    batch_fidseq_proc_thread_.join();
  }
  auto n_batch_bfidseq_ptr = parse_all_fidseq(all_chan_recs, slot_is_dense);
  batch_fidseq_chan_ = paddle::framework::MakeChannel<std::shared_ptr<BatchFidSeq>>();
  batch_fidseq_chan_->SetBlockSize(1);
  batch_fidseq_chan_->SetCapacity(2);

  batch_fidseq_proc_thread_ = std::thread([&, n_batch_bfidseq_ptr, this]() {
    auto & n_batch_bfidseq = *n_batch_bfidseq_ptr;
    for (size_t i = 0; i < n_batch_bfidseq.size(); i++) {
      auto seq = std::make_shared<BatchFidSeq>();
      seq->h_fidseq = std::move(*(n_batch_bfidseq[i]));
      seq->h_bucket_sizes.resize(worker_num_);
 
      auto & fidseq = seq->h_fidseq;

      ThreadBarrier partition_barrier(thread_num_);
      ThreadBarrier resize_barrier(thread_num_);

      std::mutex mtx;
      std::vector<std::thread> threads(thread_num_);
      std::vector<std::vector<uint32_t>> h_fid_buckets(worker_num_);
      for (int t = 0; t < thread_num_; t++) {
        threads[t] = std::thread([&, t, this] () {
          // do partition
          std::vector<std::vector<uint32_t>> thread_buckets(worker_num_);
          for (int j = t; j < (int)fidseq.size(); j += thread_num_) {
            int bucket_idx = fidseq[j] % worker_num_;
            thread_buckets[bucket_idx].push_back(fidseq[j]);
          }
          mtx.lock();
          for (int j = 0; j < worker_num_; j++) {
            h_fid_buckets[j].insert(h_fid_buckets[j].end(), thread_buckets[j].begin(), thread_buckets[j].end());
          }
          mtx.unlock();
          partition_barrier.wait();

          // merge bucket
          int max_bucket_size = 0;
          for (int j = 0; j < worker_num_; j++) {
            if ((int)h_fid_buckets[j].size() > max_bucket_size) {
              max_bucket_size = h_fid_buckets[j].size();
            }
          }
          if (t == 0) {
            seq->max_bucket_size = max_bucket_size;
          }

          if (t < (int)h_fid_buckets.size()) {
            std::sort(h_fid_buckets[t].begin(), h_fid_buckets[t].end());
          }

          if (t == 0) {
            seq->h_fidseq_bucket.resize(max_bucket_size * worker_num_, 0);
            for (int j = 0; j < worker_num_; j++) {
              seq->h_bucket_sizes[j] = h_fid_buckets[j].size();
            }
          }
          resize_barrier.wait();
          if (t < (int)h_fid_buckets.size()) {
            memcpy(&(seq->h_fidseq_bucket[0]) + t * max_bucket_size, 
                                      &(h_fid_buckets[t][0]), 
                  sizeof(uint32_t) * h_fid_buckets[t].size());
          }
        });
      }

      for (auto & thd : threads) {
        thd.join();
      }
      batch_fidseq_chan_->Put(seq);
    }
    batch_fidseq_chan_->Close();
  });

  timeline.Pause();
  VLOG(0) << "CacheManager::build_batch_fidseq:" << timeline.ElapsedSec() << "s";
}

void CacheManager::prepare_next_batch(int worker_id) {
  //platform::Timer timeline;
  //std::stringstream time_ss;
  //double total_time = 0.0;

  if (prepare_merge_grad_threads_[worker_id].joinable()) {
    prepare_merge_grad_threads_[worker_id].join();
  }

  if (FLAGS_dump_cache_manager &&
                worker_id == 0 &&
      debug_data_chan_ != nullptr &&
      current_batch_fidseq_ != nullptr) {
    //timeline.Start();
    std::string tmp_data = current_batch_fidseq_->to_string();
    debug_data_chan_->Put(std::move(tmp_data));
    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << "worker0-debug:" << timeline.ElapsedSec();
  }

  if (worker_id == 0) {
    //timeline.Start();
    //time_ss << "batch_fidseq_chan_.size:" << batch_fidseq_chan_->Size();
    if (!batch_fidseq_chan_->Get(current_batch_fidseq_)) {
      current_batch_fidseq_ = nullptr;
    }
    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",worker0-getchan:" << timeline.ElapsedSec();
 
    //timeline.Start();

    PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
    current_batch_fidseq_->d_fidseqs.resize(worker_num_, nullptr);
    current_batch_fidseq_->d_bucket_sizes.resize(worker_num_, nullptr);
    current_batch_fidseq_->d_fidseq_buckets.resize(worker_num_, nullptr);

    current_batch_fidseq_->h_cache_bfid_sizes.resize(worker_num_, 0);
    current_batch_fidseq_->d_cache_bfids.resize(worker_num_, nullptr);

    current_batch_fidseq_->h_cache_bfid_resort_indexes.resize(worker_num_);
    current_batch_fidseq_->d_cache_bfid_resort_indexes.resize(worker_num_, nullptr);
    current_batch_fidseq_->h_cache_bfid_lods.resize(worker_num_);
    current_batch_fidseq_->d_cache_bfid_lods.resize(worker_num_, nullptr);

    if (FLAGS_dump_cache_manager) { 
      current_batch_fidseq_->debug_h_cache_bfids.resize(worker_num_);
      current_batch_fidseq_->debug_h_cache_fids.resize(worker_num_);
    }

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",worker0-resize:" << timeline.ElapsedSec();
  }

  //timeline.Start();
  worker_barrier_.wait();
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",wait:" << timeline.ElapsedSec();
  
  //timeline.Start();
  int dev_id = resource_->dev_id(worker_id);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  
  current_batch_fidseq_->d_fidseqs[worker_id] = memory::Alloc(place, 
          current_batch_fidseq_->h_fidseq.size() * sizeof(uint32_t));
  current_batch_fidseq_->d_bucket_sizes[worker_id] = memory::Alloc(place,
          current_batch_fidseq_->h_bucket_sizes.size() * sizeof(uint32_t));
  current_batch_fidseq_->d_fidseq_buckets[worker_id] = memory::Alloc(place,
          current_batch_fidseq_->h_fidseq_bucket.size() * sizeof(uint32_t));

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",alloc_memory:" << timeline.ElapsedSec();

  //timeline.Start();
  auto cpu_place = platform::CPUPlace();
  //auto & stream = comm_streams_[worker_id];
  memory::Copy(place, 
              reinterpret_cast<uint32_t*>(
                  current_batch_fidseq_->d_fidseqs[worker_id]->ptr()), 
              cpu_place,
              &(current_batch_fidseq_->h_fidseq[0]),
              current_batch_fidseq_->h_fidseq.size() * sizeof(uint32_t));
  memory::Copy(place,
             reinterpret_cast<uint32_t*>(
                  current_batch_fidseq_->d_bucket_sizes[worker_id]->ptr()),
             cpu_place,
             &(current_batch_fidseq_->h_bucket_sizes[0]),
             current_batch_fidseq_->h_bucket_sizes.size() * sizeof(uint32_t));
  memory::Copy(place,
             reinterpret_cast<uint32_t*>(
                  current_batch_fidseq_->d_fidseq_buckets[worker_id]->ptr()),
             cpu_place,
             &(current_batch_fidseq_->h_fidseq_bucket[0]),
             current_batch_fidseq_->h_fidseq_bucket.size() * sizeof(uint32_t));
  xpu_wait(0);

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",memory_copy:" << timeline.ElapsedSec();
  //VLOG(0) << "CacheManager::prepare_next_batch:" << total_time << "s, detail:" << time_ss.str();
}

void CacheManager::convert_fid2bfid(int dev_id, uint32_t * fids, int fid_len) {
  //platform::Timer timeline;
  //timeline.Start();

  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  
  int worker_id = resource_->get_index_by_devid(dev_id);
  auto & stream = comm_streams_[worker_id];
  sync_stream(stream);

  current_batch_fidseq_->h_cache_bfid_sizes[worker_id] = fid_len;
  current_batch_fidseq_->d_cache_bfids[worker_id] = memory::Alloc(place, fid_len * sizeof(int));
  xpu_kernel_.convert_fid2bfid(
      reinterpret_cast<uint32_t*>(
              current_batch_fidseq_->d_fidseq_buckets[worker_id]->ptr()),
      current_batch_fidseq_->h_fidseq_bucket.size(),
      reinterpret_cast<uint32_t*>(
              current_batch_fidseq_->d_bucket_sizes[worker_id]->ptr()),
      current_batch_fidseq_->h_bucket_sizes.size(),
      fids,
      fid_len,
      reinterpret_cast<int*>(
              current_batch_fidseq_->d_cache_bfids[worker_id]->ptr()),
      stream);

  if (FLAGS_dump_cache_manager) {
    auto cpu_place = platform::CPUPlace();
    current_batch_fidseq_->debug_h_cache_fids[worker_id].resize(fid_len);
    memory::Copy(cpu_place,
             &(current_batch_fidseq_->debug_h_cache_fids[worker_id][0]),
             place,
             fids,
             fid_len * sizeof(int));
    xpu_wait(0);
  }

  //timeline.Pause();
  //VLOG(0) << "CacheManager::convert_fid2bfid:" << timeline.ElapsedSec() << "s";
}

void CacheManager::prepare_merge_grad(int dev_id) {
  int worker_id = resource_->get_index_by_devid(dev_id);
  prepare_merge_grad_threads_[worker_id] = std::thread([worker_id, dev_id, this] () {
    //platform::Timer timeline;
    //std::stringstream time_ss;
    //time_ss << "dev:" << dev_id;
    //double total_time = 0.0;
    //timeline.Start();

    PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
    DevPlace place = DevPlace(dev_id);
    AnyDeviceGuard guard(dev_id);

    int worker_id = resource_->get_index_by_devid(dev_id);
    int cache_bfid_size = current_batch_fidseq_->h_cache_bfid_sizes[worker_id];
    std::vector<int> h_cache_bfids(cache_bfid_size);
    auto cpu_place = platform::CPUPlace();
    memory::Copy(cpu_place,
        &(h_cache_bfids[0]),
        place,
        current_batch_fidseq_->d_cache_bfids[worker_id]->ptr(),
        cache_bfid_size * sizeof(int));
    xpu_wait(0);
 
    std::mutex bfid_buckets_mtx;
    std::vector<int> cache_bfid_resort_indexes(cache_bfid_size);
    std::vector<int> cache_bfid_lod(current_batch_fidseq_->h_fidseq_bucket.size() + 1, 0);
    std::vector<int> bfid_bucket_sizes(current_batch_fidseq_->h_fidseq_bucket.size(), 0);
    std::vector<std::vector<int>> bfid_buckets(current_batch_fidseq_->h_fidseq_bucket.size());

    int thread_num = 24;
    std::vector<int> bfid_counts(thread_num, 0);
    std::vector<std::thread> resort_threads(thread_num);
    ThreadBarrier resort_barrier(thread_num);
    for (int i = 0; i < thread_num; i++) {
      resort_threads[i] = std::thread([&, this] (int tid) {
        int mean_thread_data_size = (cache_bfid_size / thread_num) + 
                        ((cache_bfid_size % thread_num) > 0 ? 1 : 0);
        int begin_offset = mean_thread_data_size * tid;
        int end_offset = mean_thread_data_size * (tid + 1);
        end_offset = end_offset <= cache_bfid_size ? end_offset : cache_bfid_size;

        std::unordered_map<int, std::vector<int>> thread_bfid_buckets;
        for (int j = begin_offset; j < end_offset; j++) {
          if (h_cache_bfids[j] < 0) {
            continue;
          }
          if (thread_bfid_buckets.find(h_cache_bfids[j]) == thread_bfid_buckets.end()) {
            thread_bfid_buckets[h_cache_bfids[j]] = std::vector<int>();
          }
          thread_bfid_buckets[h_cache_bfids[j]].push_back(j);
        }
        bfid_buckets_mtx.lock();
        for (auto iter = thread_bfid_buckets.begin(); iter != thread_bfid_buckets.end(); iter++) {
          bfid_buckets[iter->first].insert(
                    bfid_buckets[iter->first].end(), iter->second.begin(), iter->second.end());
        }
        bfid_buckets_mtx.unlock();
        resort_barrier.wait();

        mean_thread_data_size = (bfid_buckets.size() / thread_num) +
                    ((bfid_buckets.size() % thread_num) > 0 ? 1 : 0);
        begin_offset = mean_thread_data_size * tid;
        end_offset = mean_thread_data_size * (tid + 1);
        end_offset = end_offset <= (int)bfid_buckets.size() ? end_offset : (int)bfid_buckets.size();
        bfid_counts[tid] = 0;
        for (int j = begin_offset; j < end_offset; j++) {
          bfid_counts[tid] += bfid_buckets[j].size();
          bfid_bucket_sizes[j] = bfid_buckets[j].size();
        }
        resort_barrier.wait();

        int copy_begin_pos = 0;
        for (int j = 0; j < tid; j++) {
          copy_begin_pos += bfid_counts[j];
        }
        
        for (int j = begin_offset; j < end_offset; j++) {
          if (j == begin_offset) {
            cache_bfid_lod[j+1] = copy_begin_pos + bfid_buckets[j].size();
          } else {
            cache_bfid_lod[j+1] = cache_bfid_lod[j] + bfid_buckets[j].size();
          }
        }

        int total_copy_size = 0;
        for (int j = begin_offset; j < end_offset; j++) {
          int copy_size = bfid_buckets[j].size();
          PADDLE_ENFORCE_LE(copy_begin_pos + total_copy_size + copy_size, cache_bfid_resort_indexes.size(),
                                                         platform::errors::External("may error hadppened"));
          memcpy(&(cache_bfid_resort_indexes[0]) + copy_begin_pos + total_copy_size, 
                                     &(bfid_buckets[j][0]), copy_size * sizeof(int));
          total_copy_size += copy_size;
        }
        //VLOG(0) << "prepare_merge_grad, dev:" << dev_id
        //        << ", thread:" << tid
        //        << ", offset:[" << begin_offset << "," << end_offset << ")" 
        //        << ", len:" << (end_offset - begin_offset) << "/" << bfid_buckets.size()
        //        << ", buffer_size:" << cache_bfid_resort_indexes.size()
        //        << ", total_copy:" << total_copy_size;
      }, i);
    }
    for (auto & thrd : resort_threads) {
      thrd.join();
    }
  
    current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id] = memory::Alloc(place, 
                                          cache_bfid_resort_indexes.size() * sizeof(int));
    memory::Copy(place,
        current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id]->ptr(),
        cpu_place,
        &(cache_bfid_resort_indexes[0]),
        cache_bfid_resort_indexes.size() * sizeof(int));
    current_batch_fidseq_->d_cache_bfid_lods[worker_id] = memory::Alloc(place,
                                                 cache_bfid_lod.size() * sizeof(int));
    memory::Copy(place,
        current_batch_fidseq_->d_cache_bfid_lods[worker_id]->ptr(),
        cpu_place,
        &(cache_bfid_lod[0]),
        cache_bfid_lod.size() * sizeof(int));
    current_batch_fidseq_->h_cache_bfid_resort_indexes[worker_id] = std::move(cache_bfid_resort_indexes);
    current_batch_fidseq_->h_cache_bfid_lods[worker_id] = std::move(cache_bfid_lod);
    xpu_wait(0);

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",prepare_merge_grad:" << timeline.ElapsedSec();
    //VLOG(0) << "prepare_merge_grad time cost:" << total_time
    //      << " sec, detail:" << time_ss.str();

    //VLOG(0) << "prepare_merge_grad, dev:" << dev_id
    //        << ", h_cache_bfid_resort_indexes:" << current_batch_fidseq_->h_cache_bfid_resort_indexes[worker_id].size()
    //        << ", h_cache_bfid_lod:" << current_batch_fidseq_->h_cache_bfid_lods[worker_id].size()
    //        << ", h_cache_bfid_size:" << current_batch_fidseq_->h_cache_bfid_sizes[worker_id];
  });
}

void CacheManager::get_merge_grad_params(int dev_id,
      int ** key_resort_idxs, int * out_key_resort_idx_len,
                   int ** fidseq_lods, int * fidseq_lod_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);

  if (prepare_merge_grad_threads_[worker_id].joinable()) {
    prepare_merge_grad_threads_[worker_id].join();
  }

  *key_resort_idxs = reinterpret_cast<int*>(
     current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id]->ptr());
  *out_key_resort_idx_len = current_batch_fidseq_->h_cache_bfid_resort_indexes[worker_id].size();
  *fidseq_lods = reinterpret_cast<int*>(
     current_batch_fidseq_->d_cache_bfid_lods[worker_id]->ptr());
  *fidseq_lod_len = current_batch_fidseq_->h_cache_bfid_lods[worker_id].size();
}

void CacheManager::get_device_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  auto d_fidseq_bucket_ptr = reinterpret_cast<uint32_t*>(
      current_batch_fidseq_->d_fidseq_buckets[worker_id]->ptr());
  *out_keys = d_fidseq_bucket_ptr + worker_id * current_batch_fidseq_->max_bucket_size;
  *out_key_len = current_batch_fidseq_->h_bucket_sizes[worker_id];
}

void CacheManager::get_device_all_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_keys = reinterpret_cast<uint32_t*>(
      current_batch_fidseq_->d_fidseq_buckets[worker_id]->ptr());
  *out_key_len = current_batch_fidseq_->h_fidseq_bucket.size();
}

const std::vector<uint32_t> & CacheManager::get_host_all_fidseq_bucket_sizes() {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  return current_batch_fidseq_->h_bucket_sizes;
}

void CacheManager::get_device_all_fidseq_bucket_sizes(int dev_id, uint32_t ** out_buffer, int * out_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_buffer = reinterpret_cast<uint32_t*>(
      current_batch_fidseq_->d_bucket_sizes[worker_id]->ptr());
  *out_len = current_batch_fidseq_->h_bucket_sizes.size();
}

void CacheManager::get_device_all_fidseq(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_keys = reinterpret_cast<uint32_t*>(
      current_batch_fidseq_->d_fidseqs[worker_id]->ptr());
  *out_key_len = current_batch_fidseq_->h_fidseq.size();
}

void CacheManager::get_bfidseq(int dev_id, int ** out_keys, int * out_key_len) {
  //platform::Timer timeline;
  //timeline.Start();

  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  auto & stream = comm_streams_[worker_id];
  sync_stream(stream);

  *out_keys = reinterpret_cast<int*>(
      current_batch_fidseq_->d_cache_bfids[worker_id]->ptr());
  *out_key_len = current_batch_fidseq_->h_cache_bfid_sizes[worker_id];

  if (FLAGS_dump_cache_manager && 
      current_batch_fidseq_->debug_h_cache_bfids[worker_id].size() == 0) {
    DevPlace place = DevPlace(dev_id);
    AnyDeviceGuard guard(dev_id);
    auto cpu_place = platform::CPUPlace();

    current_batch_fidseq_->debug_h_cache_bfids[worker_id].resize(*out_key_len);
    memory::Copy(cpu_place,
             &(current_batch_fidseq_->debug_h_cache_bfids[worker_id][0]),
             place,
             *out_keys,
             (*out_key_len) * sizeof(int));
    xpu_wait(0);
  }

  //timeline.Pause();
  //VLOG(0) << "CacheManager::get_bfidseq:" << timeline.ElapsedSec() << "s";
}

int CacheManager::get_device_bucket_mean_len() {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  return current_batch_fidseq_->max_bucket_size;
}

void CacheManager::compress_bucket(int dev_id, void * vals, int val_len, int type_size, const XPUStream & stream) {
  //platform::Timer timeline;
  //timeline.Start();

  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);

  //int worker_id = resource_->get_index_by_devid(dev_id);
  //auto & stream = comm_streams_[worker_id];
  sync_stream(stream);
 
  //VLOG(0) << "compress_bucket: h_fidseq.size():" << current_batch_fidseq_->h_fidseq.size()
  //        << ", h_fidseq_bucket.size():" << current_batch_fidseq_->h_fidseq_bucket.size();
  int buffer_size = current_batch_fidseq_->h_fidseq.size() * type_size;
  std::unique_ptr<char[]> h_val_buffer(new char[buffer_size]);
  
  auto cpu_place = platform::CPUPlace();
  int buffer_offset = 0;
  auto & bucket_sizes = current_batch_fidseq_->h_bucket_sizes;
  for (int i = 0; i < (int)bucket_sizes.size(); i++) {
    //VLOG(0) << "compress_bucket dev_id:" << dev_id << ", compress_bucket:" << i << ":" << bucket_sizes[i];
    int copy_size = bucket_sizes[i] * type_size;
    memory::Copy(cpu_place,
                &(h_val_buffer[0]) + buffer_offset,
                place,
                (char *)vals + i * (current_batch_fidseq_->max_bucket_size) * type_size,
                copy_size);
    buffer_offset += copy_size;
  }
  PADDLE_ENFORCE_EQ(buffer_offset, buffer_size);
  memory::Copy(place, vals, cpu_place, &(h_val_buffer[0]), buffer_size);
  xpu_wait(0);

  //timeline.Pause();
  //VLOG(0) << "CacheManager::compress_bucket:" << timeline.ElapsedSec() << "s";
}

#endif
}  // end namespace framework
}  // end namespace paddle

#endif
