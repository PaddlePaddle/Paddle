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
#include <vector>

#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_base.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#endif

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
class HeterPs : public HeterPsBase {
 public:
  HeterPs() {}
  HeterPs(size_t capacity,
          std::shared_ptr<HeterPsResource> resource,
          GPUAccessor& gpu_accessor);  // NOLINT
  virtual ~HeterPs();
  HeterPs(const HeterPs&) = delete;
  HeterPs& operator=(const HeterPs&) = delete;

  void pull_sparse(int num,
                   FeatureKey* d_keys,
                   float* d_vals,
                   size_t len) override;
  void build_ps(int num,
                FeatureKey* h_keys,
                char* pool,
                size_t len,
                size_t feature_value_size,
                size_t chunk_size,
                int stream_num) override;
#if defined(PADDLE_WITH_CUDA)
  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size,
                              int rank_id) override;
  void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) override;

#endif

  void set_sparse_sgd(const OptimizerConfig& optimizer_config) override;
  void set_embedx_sgd(const OptimizerConfig& optimizer_config) override;

  void end_pass() override;
  int get_index_by_devid(int devid) override;
  void show_one_table(int gpu_num) override;
  void push_sparse(int num, FeatureKey* d_keys, float* d_grads, size_t len);
  void show_table_collisions() override;
#if defined(PADDLE_WITH_CUDA)
  // dedup
  int dedup_keys_and_fillidx(const int gpu_id,
                             const int total_fea_num,
                             const FeatureKey* d_keys,   // input
                             FeatureKey* d_merged_keys,  // output
                             FeatureKey* d_sorted_keys,
                             uint32_t* d_restore_idx,
                             uint32_t* d_sorted_idx,
                             uint32_t* d_offset,
                             uint32_t* d_merged_cnts,
                             bool filter_zero);
#endif
  // reset table
  void reset_table(const int dev_id,
                   size_t capacity,
                   const OptimizerConfig& sgd_config,
                   const OptimizerConfig& embedx_config,
                   bool infer_mode) {
    comm_->reset_table(dev_id, capacity, sgd_config, embedx_config, infer_mode);
  }
  void set_mode(bool infer_mode) { comm_->set_mode(infer_mode); }

 private:
  std::shared_ptr<HeterComm<FeatureKey, float*, float*, GPUAccessor>> comm_;
#if defined(PADDLE_WITH_CUDA)
  GPUOptimizer<GPUAccessor> opt_;
#endif
};

}  // namespace framework
}  // namespace paddle
#endif
