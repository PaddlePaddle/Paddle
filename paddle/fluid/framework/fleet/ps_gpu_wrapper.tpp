// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_PSLIB
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/timer.h"


namespace paddle {
namespace framework {

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;

void PSGPUWrapper::BuildGPUPS(uint64_t table_id, int feature_dim) {
  auto fleet_ptr = FleetWrapper::GetInstance();
  if (local_tables_.find(table_id) == local_tables_.end()) {
    return;
  }
  int shard_num = local_tables_[table_id].size();
  if (shard_num == 0) {
    return;
  }
  platform::Timer timeline;
  std::vector<std::thread> threads(shard_num);
  auto ptl_func = [this, &table_id, &fleet_ptr](int i) {
    size_t key_size = this->local_tables_[table_id][i].size();
    std::vector<uint64_t> keys;
    keys.reserve(key_size);
    std::vector<float*> pull_result_ptr;
    pull_result_ptr.reserve(key_size);

    for (auto& kv : this->local_tables_[table_id][i]) {
      keys.emplace_back(kv.first);
      pull_result_ptr.emplace_back(kv.second.data());
    }
    auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse(
        pull_result_ptr.data(), table_id, keys.data(), key_size);
    tt.wait();
    auto status = tt.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << pull_result_ptr.size();
    }
  };
  timeline.Start();
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(ptl_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  // build GPUPSTask to build PS
  GpuTask_.reset(new GpuTask());
  GpuTask_->BuildTask(table_id, local_tables_[table_id]);
  GpuPs_->build_ps(table_id, GpuTask_->feature_keys_[0].data(), GpuTask_->feature_values_[0].data(), GpuTask_->size(), 10000, 2);
  
  
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                            const int table_id,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size) {
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();
  int64_t total_length = std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu =
      reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
    LoDTensor& total_keys_tensor = keys_tensor[device_id];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::AllocShared(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS";
    pull_gpups_timer.Start();
    GpuPs_->pull_sparse(table_id, total_keys, total_values_gpu,
                              static_cast<int>(total_length));
    //PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
    //                              "PullSparseGPU failed in GPUPS."));
    pull_gpups_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
  
  
}

void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                            const int table_id,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<const float*>& grad_values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size, const int batch_size) {
  VLOG(3) << "Begin GPUPS PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length * sizeof(FeaturePushValue));
  FeaturePushValue* total_grad_values_gpu = reinterpret_cast<FeaturePushValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (platform::is_gpu_place(place)) {
    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
    LoDTensor& cached_total_keys_tensor = keys_tensor[device_id];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to gpups struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in GPUPS";
    push_gpups_timer.Start();
    GpuPs_->push_sparse(table_id,
        total_keys, total_grad_values_gpu, static_cast<int>(total_length), opt_);
    push_gpups_timer.Pause();
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}


}  // end namespace framework
}  // end namespace paddle
#endif
