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

namespace paddle {
namespace framework {

HeterPsBase* HeterPsBase::get_instance(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    CommonFeatureValueAccessor feature_value_accessor,
    int optimizer_type) {
  return new HeterPs(
      capacity, resource, feature_value_accessor, optimizer_type);
}

HeterPs::HeterPs(size_t capacity,
                 std::shared_ptr<HeterPsResource> resource,
                 CommonFeatureValueAccessor feature_value_accessor,
                 int optimizer_type) {
  comm_ = std::make_shared<HeterComm<FeatureKey, float*, float*>>(capacity,
                                                                  resource);
  feature_value_accessor_ = feature_value_accessor;
  optimizer_type_ = optimizer_type;
}

HeterPs::~HeterPs() {}

void HeterPs::pull_sparse(int num,
                          FeatureKey* d_keys,
                          float* d_vals,
                          size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

int HeterPs::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

void HeterPs::set_sparse_sgd(const OptimizerConfig& optimizer_config) {
  comm_->set_sparse_sgd(optimizer_config);
}

void HeterPs::set_embedx_sgd(const OptimizerConfig& optimizer_config) {
  comm_->set_embedx_sgd(optimizer_config);
}

void HeterPs::end_pass() { comm_->end_pass(); }

void HeterPs::show_one_table(int gpu_num) { comm_->show_one_table(gpu_num); }

void HeterPs::push_sparse(int num,
                          FeatureKey* d_keys,
                          float* d_grads,
                          size_t len) {
  comm_->push_sparse(num, d_keys, d_grads, len);
  // comm_->push_sparse_multi_node(num, d_keys, d_grads, len, opt_);
}

}  // end namespace framework
}  // end namespace paddle
#endif
