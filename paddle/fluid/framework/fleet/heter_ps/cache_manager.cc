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
#include <cmath>
#include <thread>
#include <fstream>
#include <bitset>
#include <chrono>
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

std::shared_ptr<BatchFidSeq> CacheManager::parse_uniq_fids(
     const std::vector<std::deque<Record>::iterator> & train_data_iters, 
           int iter_offset, int batch_sz, const std::vector<bool> & slot_is_dense) {
  // caculate max slot & fid
  uint32_t max_fid = 0;
  int max_slot = 0;
  for (size_t train_data_idx = 0; train_data_idx < train_data_iters.size(); ++train_data_idx) {
    auto it = train_data_iters[train_data_idx] + iter_offset;
    for (int j = 0; j < batch_sz; ++j) {
      const Record & cur_rec = *(it + j);
      for (auto & fea : cur_rec.uint64_feasigns_) {
        max_fid = max_fid > fea.sign().uint64_feasign_ ? max_fid : fea.sign().uint64_feasign_;
        max_slot = max_slot > fea.slot() ? max_slot : fea.slot();
        PADDLE_ENFORCE_GE(fea.slot(), 0);
      }
    }
  }

  // set bitset of fid & slot
  PADDLE_ENFORCE_LT(max_slot, (int)slot_is_dense.size());
  int uint64_fid_bit_vec_size = (max_fid + 63) / 64;
  std::vector<std::bitset<64>> fid_bit_vec(uint64_fid_bit_vec_size);
  int uint64_slot_bit_vec_size = (max_slot + 63) / 64;
  std::vector<std::bitset<64>> slot_bit_vec(uint64_slot_bit_vec_size);
  for (size_t train_data_idx = 0; train_data_idx < train_data_iters.size(); ++train_data_idx) {
    auto it = train_data_iters[train_data_idx] + iter_offset;
    for (int j = 0; j < batch_sz; ++j) {
      const Record & cur_rec = *(it++);
      for (auto & fea : cur_rec.uint64_feasigns_) {
        if (slot_is_dense[fea.slot()]) {
          continue;
        }
        int vec_idx = fea.sign().uint64_feasign_ / 64;
        int bit_offset = fea.sign().uint64_feasign_ % 64;
        fid_bit_vec[vec_idx].set(bit_offset);

        vec_idx = fea.slot() / 64;
        bit_offset = fea.slot() % 64;
        slot_bit_vec[vec_idx].set(bit_offset);
      }
    }
  }

  // count total slots
  uint32_t slot_has_value_count = 0;
  for (auto & bs : slot_bit_vec) {
    slot_has_value_count += bs.count();
  }
  // add default 0 if slot has no fid
  if (slot_has_value_count < slot_is_dense.size()) {
    fid_bit_vec[0].set(0);
  }
  // count total fids
  int total_fid_count = 0;
  for (auto & bs : fid_bit_vec) {
    total_fid_count += bs.count();
  }

  // create BatchFidSeq
  auto seq = std::make_shared<BatchFidSeq>();
  seq->h_bucket_sizes.resize(worker_num_);

  // dump fidseq
  auto & fidseq = seq->h_fidseq;
  fidseq.resize(total_fid_count);
  uint32_t offset = 0;
  for (size_t j = 0; j < fid_bit_vec.size(); ++j) {
    uint64_t val = fid_bit_vec[j].to_ulong();
    while (val) {
      uint64_t next_val = (val - 1) & val;
      uint64_t pos_val = val - next_val;
      fidseq[offset++] = (uint64_t)log2(pos_val) + j * 64;
      val = next_val;
    }
  }
  PADDLE_ENFORCE_EQ(offset, total_fid_count);

  // count bucket size
  std::vector<int> fidseq_buckets_count(worker_num_, 0);
  for (int j = 0; j < (int)fidseq.size(); ++j) {
    int bucket_idx = fidseq[j] % worker_num_;
    ++fidseq_buckets_count[bucket_idx];
  }

  // find max bucket size
  int max_bucket_size = 0;
  for (int j = 0; j < worker_num_; ++j) {
    max_bucket_size = max_bucket_size < fidseq_buckets_count[j] ? fidseq_buckets_count[j] : max_bucket_size;
    seq->h_bucket_sizes[j] = fidseq_buckets_count[j];
    fidseq_buckets_count[j] = 0;
  }
  seq->max_bucket_size = max_bucket_size;

  // make buckets
  int total_fedseq_bucket_size = max_bucket_size * worker_num_;
  seq->h_fidseq_bucket.resize(total_fedseq_bucket_size, 0);
  for (int j = 0; j < (int)fidseq.size(); ++j) {
    int bucket_idx = fidseq[j] % worker_num_;
    int offset = bucket_idx * max_bucket_size + fidseq_buckets_count[bucket_idx]++;
    seq->h_fidseq_bucket[offset] = fidseq[j];
  }

  return seq;
}

void CacheManager::build_batch_fidseq(std::vector<std::deque<Record> *> & all_chan_recs,
                                            const std::vector<bool> & slot_is_dense) {
  platform::Timer timeline;
  std::stringstream time_ss;
  double total_time = 0.0;
  timeline.Start();

  if (build_fidseq_thread_.joinable()) {
      build_fidseq_thread_.join();
  }

  PADDLE_ENFORCE_GT(all_chan_recs.size(), 0,
      platform::errors::External("all_chan_recs size error"));

  size_t expected_chan_size = all_chan_recs[0]->size();
  for (auto & chan_recs : all_chan_recs) {
    PADDLE_ENFORCE_EQ(chan_recs->size(), expected_chan_size);
  }
 
  fidseq_chan_ = paddle::framework::MakeChannel<std::shared_ptr<BatchFidSeq>>();
 
  build_fidseq_thread_ = std::thread([all_chan_recs, slot_is_dense, this] () {
    int batch_sz = batch_sz_;
    int size = all_chan_recs[0]->size();

    std::vector<std::deque<Record>::iterator> train_data_iters(all_chan_recs.size());
    for (size_t i = 0; i < all_chan_recs.size(); ++i) {
      train_data_iters[i] = all_chan_recs[i]->begin();
    }

    ParallelThreadPool build_fidseq_pool(12);
    int thread_num = build_fidseq_pool.get_thread_num();
    fidseq_chan_->SetBlockSize(1);
    fidseq_chan_->SetCapacity(3 * thread_num);
    //ThreadBarrier barrier(thread_num);
    //std::vector<std::shared_ptr<BatchFidSeq>> result_vec(thread_num, nullptr); 

    std::mutex cv_mtx;
    std::condition_variable write_cv;

    std::atomic<int> writer_idx{0};
    build_fidseq_pool.set_task([&, this] (int tid) {
      int current_batch_sz = 0;
      for (int i = tid * batch_sz_; i < size; i += thread_num * batch_sz_) {
        current_batch_sz = std::min(batch_sz, size - i);  
        auto batch_fidseq = parse_uniq_fids(train_data_iters, i, current_batch_sz, slot_is_dense);

        std::shared_ptr<std::vector<uint32_t>> fidseq_before_opt = std::make_shared<std::vector<uint32_t>>();
        { // before opt
          // process batch data for every chan_recs
          std::set<uint32_t> current_bfid_set;
          std::set<int> slot_has_val;
          for (size_t train_data_idx = 0; train_data_idx < train_data_iters.size(); ++train_data_idx) {
            auto it = train_data_iters[train_data_idx] + i;
            for (int j = 0; j < current_batch_sz; ++j) {
              const Record & cur_rec = *(it++);
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
          fidseq_before_opt->assign(current_bfid_set.begin(), current_bfid_set.end());

          PADDLE_ENFORCE_EQ(batch_fidseq->h_fidseq.size(), fidseq_before_opt->size());
          for (size_t j = 0; j < batch_fidseq->h_fidseq.size(); ++j) {
            PADDLE_ENFORCE_EQ(batch_fidseq->h_fidseq[j], (*fidseq_before_opt)[j]);
          }
          //VLOG(0) << "tid:" << tid << ", check consistency ok";
        }
        while (true) {
          std::unique_lock<std::mutex> lock(cv_mtx);
          if (tid == writer_idx.load()) {
            fidseq_chan_->Put(std::move(batch_fidseq));
            writer_idx.store((tid + 1) % thread_num);
            write_cv.notify_all();
            break;
          }
          write_cv.wait_for(lock, std::chrono::milliseconds(30));
        }
      }
    });
    build_fidseq_pool.wait_task();
    fidseq_chan_->Close();
  });

  timeline.Pause();
  total_time += timeline.ElapsedSec();
  time_ss << "lauch-async-thread:" << timeline.ElapsedSec();
  timeline.Start();

  while (!fidseq_chan_->Closed() && fidseq_chan_->Size() == 0) { }
  timeline.Pause();
  total_time += timeline.ElapsedSec();
  time_ss << ",wait-data-ready:" << timeline.ElapsedSec();
  VLOG(0) << "build_batch_fidseq total_time:" << total_time << "s, details:" << time_ss.str();
}

void CacheManager::prepare_next_batch(int worker_id) {
  //platform::Timer timeline;
  //std::stringstream time_ss;
  //double total_time = 0.0;
  //timeline.Start();

  if (prepare_merge_grad_threads_[worker_id].joinable()) {
    prepare_merge_grad_threads_[worker_id].join();
  }
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << "join-prepare_merge_grad_threads:" << timeline.ElapsedSec();

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
    //time_ss << ",fidseq_chan_.size:" << fidseq_chan_->Size();
    while (!fidseq_chan_->Closed() && fidseq_chan_->Size() < 2) { }
    if (!fidseq_chan_->Get(current_batch_fidseq_)) {
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

  current_batch_fidseq_->d_fidseqs[worker_id] =
      malloc_l3_or_gm<uint32_t>(current_batch_fidseq_->h_bucket_sizes.size(), dev_id);
  current_batch_fidseq_->d_bucket_sizes[worker_id] =
      malloc_l3_or_gm<uint32_t>(current_batch_fidseq_->h_bucket_sizes.size(), dev_id);
  current_batch_fidseq_->d_fidseq_buckets[worker_id] =
      malloc_l3_or_gm<uint32_t>(current_batch_fidseq_->h_fidseq_bucket.size(), dev_id);

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",alloc_memory:" << timeline.ElapsedSec();

  //timeline.Start();
  auto cpu_place = platform::CPUPlace();
  //auto & stream = comm_streams_[worker_id];
  memory::Copy(place,
              current_batch_fidseq_->d_fidseqs[worker_id].get(),
              cpu_place,
              &(current_batch_fidseq_->h_fidseq[0]),
              current_batch_fidseq_->h_fidseq.size() * sizeof(uint32_t));
  memory::Copy(place,
             current_batch_fidseq_->d_bucket_sizes[worker_id].get(),
             cpu_place,
             &(current_batch_fidseq_->h_bucket_sizes[0]),
             current_batch_fidseq_->h_bucket_sizes.size() * sizeof(uint32_t));
  memory::Copy(place,
             current_batch_fidseq_->d_fidseq_buckets[worker_id].get(),
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
  // platform::Timer timeline;
  // timeline.Start();

  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);

  int worker_id = resource_->get_index_by_devid(dev_id);
  auto & stream = comm_streams_[worker_id];
  sync_stream(stream);

  current_batch_fidseq_->h_cache_bfid_sizes[worker_id] = fid_len;
  current_batch_fidseq_->d_cache_bfids[worker_id] = malloc_l3_or_gm<int>(fid_len, dev_id);

  xpu_kernel_.convert_fid2bfid(
      current_batch_fidseq_->d_fidseq_buckets[worker_id].get(),
      current_batch_fidseq_->h_fidseq_bucket.size(),
      current_batch_fidseq_->d_bucket_sizes[worker_id].get(),
      current_batch_fidseq_->h_bucket_sizes.size(),
      fids,
      fid_len,
      current_batch_fidseq_->d_cache_bfids[worker_id].get(),
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

  // sync_stream(stream);
  // timeline.Pause();
  // VLOG(0) << "CacheManager::convert_fid2bfid:" << timeline.ElapsedSec() << "s";
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
        current_batch_fidseq_->d_cache_bfids[worker_id].get(),
        cache_bfid_size * sizeof(int));
    xpu_wait(0);

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",copy_to_cpu:" << timeline.ElapsedSec();
    //timeline.Start();

    std::vector<int> bfid_uniq_count(current_batch_fidseq_->h_fidseq_bucket.size(), 0);
    for (int i = 0; i < cache_bfid_size; i++) {
      PADDLE_ENFORCE_LT(h_cache_bfids[i], (int)bfid_uniq_count.size(),
                            platform::errors::External("error bfid"));
      bfid_uniq_count[h_cache_bfids[i]]++;
    }

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",bfid_uniq_count:" << timeline.ElapsedSec();
    //timeline.Start();

    std::vector<int> bfid_uniq_lod(bfid_uniq_count.size() + 1, 0);
    for (int i = 0; i < (int)bfid_uniq_count.size(); i++) {
      bfid_uniq_lod[i + 1] = bfid_uniq_lod[i] + bfid_uniq_count[i];
    }

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",sum-lod:" << timeline.ElapsedSec();
    //timeline.Start();

    std::vector<int> bfid_out_buffer(cache_bfid_size);
    std::vector<int> bfid_uniq_offset(current_batch_fidseq_->h_fidseq_bucket.size(), 0);
    int tmp_bfid = 0, tmp_pos = 0;
    for (int i = 0; i < cache_bfid_size; i++) {
      tmp_bfid = h_cache_bfids[i];
      tmp_pos = bfid_uniq_offset[tmp_bfid]++;
      tmp_pos += bfid_uniq_lod[tmp_bfid];
      bfid_out_buffer[tmp_pos] = i;
    }

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",sort-bfid-pairs:" << timeline.ElapsedSec();
    //timeline.Start();

    current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id] =
        malloc_l3_or_gm<int>(cache_bfid_size, dev_id);

    memory::Copy(place,
        current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id].get(),
        cpu_place,
        &(bfid_out_buffer[0]),
        cache_bfid_size * sizeof(int));

    current_batch_fidseq_->d_cache_bfid_lods[worker_id] =
        malloc_l3_or_gm<int>(bfid_uniq_lod.size(), dev_id);

    memory::Copy(place,
        current_batch_fidseq_->d_cache_bfid_lods[worker_id].get(),
        cpu_place,
        &(bfid_uniq_lod[0]),
        bfid_uniq_lod.size() * sizeof(int));

    current_batch_fidseq_->h_cache_bfid_resort_indexes[worker_id] = std::move(bfid_out_buffer);
    current_batch_fidseq_->h_cache_bfid_lods[worker_id] = std::move(bfid_uniq_lod);
    xpu_wait(0);

    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",copy_to_xpu:" << timeline.ElapsedSec();
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
                   int ** fidseq_lods, int * fidseq_lod_len, uint32_t * first_fidseq_elem) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);

  if (prepare_merge_grad_threads_[worker_id].joinable()) {
    prepare_merge_grad_threads_[worker_id].join();
  }

  *key_resort_idxs = current_batch_fidseq_->d_cache_bfid_resort_indexes[worker_id].get();
  *out_key_resort_idx_len = current_batch_fidseq_->h_cache_bfid_resort_indexes[worker_id].size();
  *fidseq_lods = current_batch_fidseq_->d_cache_bfid_lods[worker_id].get();
  *fidseq_lod_len = current_batch_fidseq_->h_cache_bfid_lods[worker_id].size();
  if (current_batch_fidseq_->h_bucket_sizes[0] > 0) {
      *first_fidseq_elem = current_batch_fidseq_->h_fidseq_bucket[0];
  }
}

void CacheManager::get_device_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  auto d_fidseq_bucket_ptr = current_batch_fidseq_->d_fidseq_buckets[worker_id].get();
  *out_keys = d_fidseq_bucket_ptr + worker_id * current_batch_fidseq_->max_bucket_size;
  *out_key_len = current_batch_fidseq_->h_bucket_sizes[worker_id];
}

void CacheManager::get_device_all_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_keys = current_batch_fidseq_->d_fidseq_buckets[worker_id].get();
  *out_key_len = current_batch_fidseq_->h_fidseq_bucket.size();
}

const std::vector<uint32_t> & CacheManager::get_host_all_fidseq_bucket_sizes() {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  return current_batch_fidseq_->h_bucket_sizes;
}

void CacheManager::get_device_all_fidseq_bucket_sizes(int dev_id, uint32_t ** out_buffer, int * out_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_buffer = current_batch_fidseq_->d_bucket_sizes[worker_id].get();
  *out_len = current_batch_fidseq_->h_bucket_sizes.size();
}

void CacheManager::get_device_all_fidseq(int dev_id, uint32_t ** out_keys, int * out_key_len) {
  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  *out_keys = current_batch_fidseq_->d_fidseqs[worker_id].get();
  *out_key_len = current_batch_fidseq_->h_fidseq.size();
}

void CacheManager::get_bfidseq(int dev_id, int ** out_keys, int * out_key_len) {
  //platform::Timer timeline;
  //timeline.Start();

  PADDLE_ENFORCE_NOT_NULL(current_batch_fidseq_);
  int worker_id = resource_->get_index_by_devid(dev_id);
  auto & stream = comm_streams_[worker_id];
  sync_stream(stream);

  *out_keys = current_batch_fidseq_->d_cache_bfids[worker_id].get();
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
