/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"

namespace paddle::platform {
namespace {
// TODO(panyx0718): Where to destroy them.
std::unique_ptr<std::vector<ncclComm_t>> global_comms;
std::unique_ptr<std::unordered_map<int, int>> comm_id_map;
bool inited = false;
size_t last_num_gpus = -1;
// TODO(panyx0718): Need to decide whether Paddle supports parallel
// runs with different number GPUs. If true, current solution is not enough.
std::mutex comm_mu;
}  // namespace

int Communicator::GetCommId(int device_id) const {
  std::lock_guard<std::mutex> guard(comm_mu);
  return comm_id_map->at(device_id);
}

void Communicator::InitAll(const std::vector<int>& gpus) {
  std::lock_guard<std::mutex> guard(comm_mu);
  if (inited && last_num_gpus == gpus.size()) {
    return;
  }
  last_num_gpus = gpus.size();
  if (global_comms) {
    for (size_t i = 0; i < global_comms->size(); ++i) {
      // FIXME(dzh) : PADDLE_ENFORCE return void
      phi::dynload::ncclCommDestroy((*global_comms)[i]);
    }
  }
  global_comms = std::make_unique<std::vector<ncclComm_t>>();
  comm_id_map = std::make_unique<std::unordered_map<int, int>>();
  global_comms->resize(gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i) {
    (*comm_id_map)[gpus[i]] = i;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclCommInitAll(
      global_comms->data(), gpus.size(), gpus.data()));
  inited = true;
}

const std::vector<ncclComm_t>& Communicator::comms() const {
  std::lock_guard<std::mutex> guard(comm_mu);
  return *global_comms;
}

}  // namespace paddle::platform
