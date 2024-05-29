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

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

class HeterPsBase {
 public:
  HeterPsBase() {}
  HeterPsBase(size_t capacity, std::shared_ptr<HeterPsResource> resource) {}
  virtual ~HeterPsBase() {}
  HeterPsBase(const HeterPsBase&) = delete;
  HeterPsBase& operator=(const HeterPsBase&) = delete;

  virtual void pull_sparse(int num,
                           FeatureKey* d_keys,
                           float* d_vals,
                           size_t len) = 0;
  virtual void build_ps(int num,
                        FeatureKey* h_keys,
                        char* pool,
                        size_t len,
                        size_t feature_value_size,
                        size_t chunk_size,
                        int stream_num) = 0;
  virtual int get_index_by_devid(int devid) = 0;
#if defined(PADDLE_WITH_CUDA)
  virtual void set_nccl_comm_and_size(
      const std::vector<ncclComm_t>& inner_comms,
      const std::vector<ncclComm_t>& inter_comms,
      int comm_size,
      int rank_id) = 0;
  virtual void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) = 0;

#endif
  virtual void end_pass() = 0;
  virtual void show_one_table(int gpu_num) = 0;
  virtual void show_table_collisions() = 0;
  virtual void push_sparse(int num,
                           FeatureKey* d_keys,
                           float* d_grads,
                           size_t len) = 0;

  virtual void set_sparse_sgd(const OptimizerConfig& optimizer_config) = 0;
  virtual void set_embedx_sgd(const OptimizerConfig& optimizer_config) = 0;

  static HeterPsBase* get_instance(
      size_t capacity,
      std::shared_ptr<HeterPsResource> resource,
      std::unordered_map<std::string, float> fleet_config,
      std::string accessor_type,
      int optimizer_type);
#if defined(PADDLE_WITH_CUDA)
  // dedup
  virtual int dedup_keys_and_fillidx(const int gpu_id,
                                     const int total_fea_num,
                                     const FeatureKey* d_keys,   // input
                                     FeatureKey* d_merged_keys,  // output
                                     FeatureKey* d_sorted_keys,
                                     uint32_t* d_restore_idx,
                                     uint32_t* d_sorted_idx,
                                     uint32_t* d_offset,
                                     uint32_t* d_merged_cnts,
                                     bool filter_zero) = 0;
#endif
  virtual void reset_table(const int dev_id,
                           size_t capacity,
                           const OptimizerConfig& sgd_config,
                           const OptimizerConfig& embedx_config,
                           bool infer_mode) = 0;
  virtual void set_mode(bool infer_mode) = 0;
};

}  // namespace framework
}  // namespace paddle
#endif
