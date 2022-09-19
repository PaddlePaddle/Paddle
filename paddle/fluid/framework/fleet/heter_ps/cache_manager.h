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
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if defined(PADDLE_WITH_XPU_KP)
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"

#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/thread_barrier.h"
#include "paddle/fluid/framework/fleet/heter_ps/cache_manager_kernel.h"
#include "paddle/fluid/framework/fleet/heter_ps/parallel_thread_pool.h"

namespace paddle {
namespace framework {
#if defined(PADDLE_WITH_XPU_CACHE_BFID)

struct BatchFidSeq {
  std::vector<uint32_t> h_fidseq;
  std::vector<std::shared_ptr<uint32_t>> d_fidseqs; // fidseq for different devices

  int max_bucket_size = 0;
  std::vector<uint32_t> h_bucket_sizes;
  std::vector<uint32_t> h_fidseq_bucket;
  std::vector<std::shared_ptr<uint32_t>> d_bucket_sizes;
  std::vector<std::shared_ptr<uint32_t>> d_fidseq_buckets; // fidseq partition bucket for different devices

  std::vector<uint32_t> h_cache_bfid_sizes;
  std::vector<std::vector<int>> debug_h_cache_bfids;
  std::vector<std::vector<int>> debug_h_cache_fids;
  std::vector<std::shared_ptr<int>> d_cache_bfids; // cache bfids in pull/push for different devices

  std::vector<std::vector<int>> h_cache_bfid_resort_indexes;
  std::vector<std::shared_ptr<int>> d_cache_bfid_resort_indexes;
  std::vector<std::vector<int>> h_cache_bfid_lods;
  std::vector<std::shared_ptr<int>> d_cache_bfid_lods;

  std::string to_string() {
    std::stringstream data_ss;
    data_ss << "BatchFidSeq begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
    data_ss << "h_fidseq start:\n";
    for (size_t i = 0; i < h_fidseq.size(); i++) {
      data_ss << "h_fidseq:" << i << " " << h_fidseq[i] << "\n";
    }

    data_ss << "\nmax_bucket_size:" << max_bucket_size << "\n\n";

    data_ss << "\nh_bucket_sizes start:\n";
    for (size_t i = 0; i < h_bucket_sizes.size(); i++) {
      data_ss << "h_bucket_sizes:" << i << " " << h_bucket_sizes[i] << "\n";
    }

    data_ss << "\nh_fidseq_bucket start:\n";
    for (size_t i = 0; i < h_fidseq_bucket.size(); i++) {
      data_ss << "h_fidseq_bucket:" << i << " " << h_fidseq_bucket[i] << "\n";
    }

    data_ss << "\ndebug_h_cache_bfids:\n";
    for (size_t i = 0; i < debug_h_cache_bfids.size(); i++) {
      for (size_t j = 0; j < debug_h_cache_bfids[i].size(); j++) {
        data_ss << "debug_h_cache_bfids-" << i << ": " << j
                << " " << debug_h_cache_fids[i][j]
                << " " << debug_h_cache_bfids[i][j] << "\n";
      }
    }

    data_ss << "\nh_cache_bfid_resort_indexes:\n";
    for (size_t d = 0; d < h_cache_bfid_lods.size(); d++) {
      for (size_t i = 1; i < h_cache_bfid_lods[d].size(); i++) {
        data_ss << "dev:" << d;
        data_ss << ", h_cache_bfid_lods-" << i << ":" << h_cache_bfid_lods[d][i] << " [";
        for (int j = h_cache_bfid_lods[d][i - 1]; j < h_cache_bfid_lods[d][i]; j++) {
          data_ss << j << ":" << h_cache_bfid_resort_indexes[d][j]
                  << "->bfid:" << debug_h_cache_bfids[d][h_cache_bfid_resort_indexes[d][j]]
                  << "->fid:" << debug_h_cache_fids[d][h_cache_bfid_resort_indexes[d][j]] << " ";
      }
      data_ss << "]\n";
    }
    }

    data_ss << "BatchFidSeq end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
    return data_ss.str();
  }
};

#endif

class CacheManager {
 public:
  struct CacheMeta {
   public:
    uint64_t sign_;
  };

  CacheManager(std::shared_ptr<HeterPsResource> resource);
  ~CacheManager() {
    if (build_fidseq_thread_.joinable()) {
      build_fidseq_thread_.join();
    }

    for (auto & thrd : prepare_merge_grad_threads_) {
      if (thrd.joinable()) {
        thrd.join();
      }
    }

    for (int i = 0; i < worker_num_; i++) {
      int dev_id = resource_->dev_id(i);
      AnyDeviceGuard guard(dev_id);
      auto & stream = comm_streams_[i];
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(stream));
    }
  }

  // Todo: need check init state before every function call
  void init(int thread_num, int batch_sz, int worker_num);

  void clear_sign2fids();
  void build_sign2fids(const FeatureKey* d_keys, size_t len);
  uint32_t query_sign2fid(const FeatureKey & key);
  uint64_t query_fid2sign(const uint32_t & fid);
  uint32_t get_max_fid();

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  std::shared_ptr<BatchFidSeq> parse_uniq_fids(
      const std::vector<std::deque<Record>::iterator> & train_data_iters,
                                           int iter_offset, int batch_sz, 
                                 const std::vector<bool> & slot_is_dense);
  void build_batch_fidseq(
      std::vector<std::deque<Record> *> & all_chan_recs,
                const std::vector<bool> & slot_is_dense);
  void prepare_next_batch(int worker_id);
  void convert_fid2bfid(int dev_id, uint32_t * fids, int fid_len);
  void get_device_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len);
  void get_device_all_fidseq_bucket(int dev_id, uint32_t ** out_keys, int * out_key_len);
  const std::vector<uint32_t> & get_host_all_fidseq_bucket_sizes();
  void get_device_all_fidseq_bucket_sizes(int dev_id, uint32_t ** out_buffer, int * out_len);
  void get_device_all_fidseq(int dev_id, uint32_t ** out_keys, int * out_key_len);
  void get_bfidseq(int dev_id, int ** out_keys, int * out_key_len);
  int get_device_bucket_mean_len();
  void compress_bucket(int dev_id, void * vals, int val_len, int type_size, const XPUStream & stream);
  template<class T>
  void compress_bucket(int dev_id, T * vals, int val_len, const XPUStream & stream) {
    compress_bucket(dev_id, vals, val_len, sizeof(T), stream);
  }

  void prepare_merge_grad(int dev_id);
  void get_merge_grad_params(int dev_id,
      int ** key_resort_idxs, int * out_key_resort_idx_len,
                   int ** fidseq_lods, int * fidseq_lod_len, uint32_t * first_fidseq_elem);

  template<typename T>
  std::shared_ptr<T> malloc_l3_or_gm(int count, int dev_id) {
      T* address;
      int ret = xpu_malloc(
            reinterpret_cast<void**>(&address),
            count * sizeof(T),
            XPU_MEM_L3);

      if (XPU_SUCCESS != ret) {
          ret = xpu_malloc(reinterpret_cast<void**>(&address),
                  count * sizeof(T));
      }

      PADDLE_ENFORCE_EQ(
          ret, XPU_SUCCESS,
          platform::errors::External(
              "XPU API return wrong value[%d], no enough memory", ret));

      auto free_func = [dev_id](T* ptr) {
          AnyDeviceGuard guard(dev_id);
          //VLOG(0) << "dev_id: " << dev_id << ", xpu_free: " << ptr;
          xpu_free(ptr);
      };

      return std::shared_ptr<T>(address, free_func);
  }
#endif

  std::string dump_to_file();

  template <typename StreamType>
  void sync_stream(const StreamType& stream) {
#if defined(PADDLE_WITH_XPU_KP)
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#endif
  }

 private:
  std::shared_ptr<HeterPsResource> resource_;
  CacheManagerKernel xpu_kernel_;

  std::vector<std::shared_ptr<std::atomic<int>>> dev_feasign_cnts_;
  std::unordered_map<FeatureKey, uint32_t> sign2fid_;
  std::vector<CacheMeta> fid2meta_;

  int thread_num_;
  int batch_sz_;
  int worker_num_;

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  // for batch fid sequence
  std::thread build_fidseq_thread_;
  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<BatchFidSeq>>> fidseq_chan_ = nullptr;

  ThreadBarrier worker_barrier_;
  std::shared_ptr<BatchFidSeq> current_batch_fidseq_ = nullptr;

  std::vector<ppStream> comm_streams_;
  std::shared_ptr<paddle::framework::ChannelObject<std::string>> debug_data_chan_ = nullptr;

  std::vector<std::thread> prepare_merge_grad_threads_;
#endif
};

}  // end namespace framework
}  // end namespace paddle
#endif
