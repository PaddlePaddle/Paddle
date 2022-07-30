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

#include <vector>

#include "paddle/fluid/framework/fleet/heter_ps/heter_ps.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

HeterPsBase* HeterPsBase::get_instance(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    std::unordered_map<std::string, float> fleet_config,
    std::string accessor_type,
    int optimizer_type) {
  if (accessor_type == "CtrDymfAccessor" &&
      (optimizer_type == 1 || optimizer_type == 3 || optimizer_type == 4)) {
    return new HeterPs<CommonFeatureValueAccessor>(
        capacity, resource, fleet_config, accessor_type, optimizer_type);
  } else {
    VLOG(0) << " HeterPsBase get_instance Warning: now only support "
               "CtrDymfAccessor, but get "
            << accessor_type;
    return new HeterPs<CommonFeatureValueAccessor>(
        capacity, resource, fleet_config, accessor_type, optimizer_type);
  }
}

template <typename FVAccessor>
HeterPs<FVAccessor>::HeterPs(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    std::unordered_map<std::string, float> fleet_config,
    std::string accessor_type,
    int optimizer_type) {
  comm_ = std::make_shared<HeterComm<FeatureKey, float*, float*, FVAccessor>>(
      capacity, resource);
  feature_value_accessor_.Configure(fleet_config);
  set_accessor(feature_value_accessor_);
  accessor_type_ = accessor_type;
  optimizer_type_ = optimizer_type;
}

template <typename FVAccessor>
HeterPs<FVAccessor>::~HeterPs() {}

template <typename FVAccessor>
void HeterPs<FVAccessor>::pull_sparse(int num,
                                      FeatureKey* d_keys,
                                      float* d_vals,
                                      size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::build_ps(int num,
                                   FeatureKey* h_keys,
                                   char* pool,
                                   size_t len,
                                   size_t feature_value_size,
                                   size_t chunk_size,
                                   int stream_num) {
  comm_->build_ps(
      num, h_keys, pool, len, feature_value_size, chunk_size, stream_num);
}

template <typename FVAccessor>
int HeterPs<FVAccessor>::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  comm_->set_sparse_sgd(optimizer_config);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  comm_->set_embedx_sgd(optimizer_config);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::end_pass() {
  comm_->end_pass();
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::show_one_table(int gpu_num) {
  comm_->show_one_table(gpu_num);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::push_sparse(int num,
                                      FeatureKey* d_keys,
                                      float* d_grads,
                                      size_t len) {
  if (accessor_type_ == "CtrDymfAccessor") {
    if (optimizer_type_ == 3) {  // adam
      auto optimizer = SparseAdamOptimizer(feature_value_accessor_);
      VLOG(5) << "INTO push_sparse SparseAdamOptimizer, EmbedDim():"
              << optimizer.EmbedDim();
      comm_->push_sparse(num, d_keys, d_grads, len, optimizer);
    } else if (optimizer_type_ == 4) {  // shared_adam
      auto optimizer = SparseAdamSharedOptimizer(feature_value_accessor_);
      VLOG(5) << "INTO push_sparse SparseAdamSharedOptimizer, EmbedDim():"
              << optimizer.EmbedDim();
      comm_->push_sparse(num, d_keys, d_grads, len, optimizer);
    } else if (optimizer_type_ == 1) {  // adagrad  {
      auto optimizer = SparseAdagradOptimizer(feature_value_accessor_);
      VLOG(5) << "INTO push_sparse SparseAdagradOptimizer, EmbedDim():"
              << optimizer.EmbedDim();
      comm_->push_sparse(num, d_keys, d_grads, len, optimizer);
    } else {
      VLOG(0) << " push sparse Error: CtrDymfAccessor only support adagrad(1),"
                 "adam(3) or shared_adam(4), bug get optimizer type:"
              << optimizer_type_;
    }
  } else {
    VLOG(0) << " push sparse Error: now only support CtrDymfAccessor, but get "
            << accessor_type_;
  }
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::set_nccl_comm_and_size(
    const std::vector<ncclComm_t>& inner_comms,
    const std::vector<ncclComm_t>& inter_comms,
    int comm_size) {
  comm_->set_nccl_comm_and_size(inner_comms, inter_comms, comm_size);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
  comm_->set_multi_mf_dim(multi_mf_dim, max_mf_dim);
}

template <typename FVAccessor>
void HeterPs<FVAccessor>::set_accessor(FVAccessor& accessor) {
  comm_->set_accessor(accessor);
}

}  // end namespace framework
}  // end namespace paddle
#endif
