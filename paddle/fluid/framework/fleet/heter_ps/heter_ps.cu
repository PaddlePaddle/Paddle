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
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  return new HeterPs(capacity, resource);
}

HeterPs::HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  comm_ =
      std::make_shared<HeterComm<FeatureKey, FeatureValue, FeaturePushValue>>(
          capacity, resource);
  opt_ = Optimizer<FeatureValue, FeaturePushValue>();
}

HeterPs::~HeterPs() {}

void HeterPs::pull_sparse(int num, FeatureKey* d_keys, FeatureValue* d_vals,
                          size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

void HeterPs::build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals,
                       size_t len, size_t chunk_size, int stream_num) {
  comm_->build_ps(num, h_keys, h_vals, len, chunk_size, stream_num);
}

int HeterPs::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

void HeterPs::end_pass() { comm_->end_pass(); }

void HeterPs::show_one_table(int gpu_num) { comm_->show_one_table(gpu_num); }

void HeterPs::push_sparse(int num, FeatureKey* d_keys,
                          FeaturePushValue* d_grads, size_t len) {
  comm_->push_sparse(num, d_keys, d_grads, len, opt_);
  // comm_->push_sparse_multi_node(num, d_keys, d_grads, len, opt_);
}

void HeterPs::set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                                     const std::vector<ncclComm_t>& inter_comms,
                                     int comm_size) {
  comm_->set_nccl_comm_and_size(inner_comms, inter_comms, comm_size);
}

}  // end namespace framework
}  // end namespace paddle
#endif
