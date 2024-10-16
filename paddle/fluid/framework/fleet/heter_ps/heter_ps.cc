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

#include "paddle/fluid/framework/fleet/heter_ps/heter_ps.h"
#include <vector>

#ifdef PADDLE_WITH_HETERPS

namespace paddle::framework {

HeterPsBase* HeterPsBase::get_instance(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    std::unordered_map<std::string, float> fleet_config,
    std::string accessor_type,
    int optimizer_type) {
  if (accessor_type == "CtrDymfAccessor") {
    auto* accessor_wrapper_ptr =
        GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
    CommonFeatureValueAccessor* gpu_accessor =
        ((AccessorWrapper<CommonFeatureValueAccessor>*)accessor_wrapper_ptr)
            ->AccessorPtr();
    if (optimizer_type == 1) {
      return new HeterPs<CommonFeatureValueAccessor, SparseAdagradOptimizer>(
          capacity, resource, *gpu_accessor);
    } else if (optimizer_type == 3) {
      return new HeterPs<CommonFeatureValueAccessor, SparseAdamOptimizer>(
          capacity, resource, *gpu_accessor);
    } else if (optimizer_type == 4) {
      return new HeterPs<CommonFeatureValueAccessor, SparseAdamSharedOptimizer>(
          capacity, resource, *gpu_accessor);
    } else if (optimizer_type == 5) {
      return new HeterPs<CommonFeatureValueAccessor, SparseAdagradV2Optimizer>(
          capacity, resource, *gpu_accessor);
    }
  } else {
    VLOG(0) << " HeterPsBase get_instance Warning: now only support "
               "CtrDymfAccessor, but get "
            << accessor_type;
    return new HeterPs<CommonFeatureValueAccessor, SparseAdagradOptimizer>(
        capacity, resource, fleet_config, accessor_type, optimizer_type);
  }
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
HeterPs<GPUAccessor, GPUOptimizer>::HeterPs(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    const GPUAccessor& gpu_accessor) {
  comm_ = std::make_shared<HeterComm<FeatureKey, float*, float*, GPUAccessor>>(
      capacity, resource);
  opt_ = GPUOptimizer<GPUAccessor>(gpu_accessor);
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
HeterPs<GPUAccessor, GPUOptimizer>::~HeterPs() {}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::pull_sparse(int num,
                                                     FeatureKey* d_keys,
                                                     float* d_vals,
                                                     size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
int HeterPs<GPUAccessor, GPUOptimizer>::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  comm_->set_sparse_sgd(optimizer_config);
}

void HeterPs<GPUAccessor, GPUOptimizer>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  comm_->set_embedx_sgd(optimizer_config);
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::end_pass() {
  comm_->end_pass();
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::show_one_table(int gpu_num) {
  comm_->show_one_table(gpu_num);
}

template <typename GPUAccessor, template <typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::push_sparse(int num,
                                                     FeatureKey* d_keys,
                                                     float* d_grads,
                                                     size_t len) {
  comm_->push_sparse(num, d_keys, d_grads, len);
}

}  // namespace paddle::framework
#endif
