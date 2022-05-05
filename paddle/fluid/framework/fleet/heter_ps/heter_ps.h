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

class HeterPs : public HeterPsBase {
 public:
  HeterPs() {}
  HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource);
  virtual ~HeterPs();
  HeterPs(const HeterPs&) = delete;
  HeterPs& operator=(const HeterPs&) = delete;

  void pull_sparse(int num, FeatureKey* d_keys, FeatureValue* d_vals,
                   size_t len) override;
  void build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals, size_t len,
                size_t chunk_size, int stream_num) override;

#if defined(PADDLE_WITH_CUDA)
  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size) override;
#endif

#if defined(PADDLE_WITH_XPU_KP)
  void set_sparse_sgd(const OptimizerConfig& optimizer_config) override;
  void set_embedx_sgd(const OptimizerConfig& optimizer_config) override;
#endif

  void end_pass() override;
  int get_index_by_devid(int devid) override;
  void show_one_table(int gpu_num) override;
  void push_sparse(int num, FeatureKey* d_keys, FeaturePushValue* d_grads,
                   size_t len) override;

 private:
  std::shared_ptr<HeterComm<FeatureKey, FeatureValue, FeaturePushValue>> comm_;
#if defined(PADDLE_WITH_CUDA)
  Optimizer<FeatureValue, FeaturePushValue> opt_;
#endif
};

}  // end namespace framework
}  // end namespace paddle
#endif
