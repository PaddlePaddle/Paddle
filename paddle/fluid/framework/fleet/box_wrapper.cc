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

#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
cudaStream_t BoxWrapper::stream_list_[8];
std::shared_ptr<boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;
AfsManager* BoxWrapper::afs_manager = nullptr;

void BasicAucCalculator::compute() {
  double* table[2] = {&_table[0][0], &_table[1][0]};

  double area = 0;
  double fp = 0;
  double tp = 0;

  for (int i = _table_size - 1; i >= 0; i--) {
    double newfp = fp + table[0][i];
    double newtp = tp + table[1][i];
    area += (newfp - fp) * (tp + newtp) / 2;
    fp = newfp;
    tp = newtp;
  }

  if (fp < 1e-3 || tp < 1e-3) {
    _auc = -0.5;  // which means all nonclick or click
  } else {
    _auc = area / (fp * tp);
  }

  _mae = _local_abserr / (fp + tp);
  _rmse = sqrt(_local_sqrerr / (fp + tp));
  _actual_ctr = tp / (fp + tp);
  _predicted_ctr = _local_pred / (fp + tp);
  _size = fp + tp;
}

void BasicAucCalculator::calculate_bucket_error() {
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  double* table[2] = {&_table[0][0], &_table[1][0]};
  for (int i = 0; i < _table_size; i++) {
    double click = table[1][i];
    double show = table[0][i] + table[1][i];
    double ctr = static_cast<double>(i) / _table_size;
    if (fabs(ctr - last_ctr) > kMaxSpan) {
      last_ctr = ctr;
      impression_sum = 0.0;
      ctr_sum = 0.0;
      click_sum = 0.0;
    }
    impression_sum += show;
    ctr_sum += ctr * show;
    click_sum += click;
    double adjust_ctr = ctr_sum / impression_sum;
    double relative_error =
        sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
    if (relative_error < kRelativeErrorBound) {
      double actual_ctr = click_sum / impression_sum;
      double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
      error_sum += relative_ctr_error * impression_sum;
      error_count += impression_sum;
      last_ctr = -1;
    }
  }
  _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
}

// Deprecated: should use BeginFeedPass & EndFeedPass
void BoxWrapper::FeedPass(int date,
                          const std::vector<uint64_t>& feasgin_to_box) const {
  int ret = boxps_ptr_->FeedPass(date, feasgin_to_box);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "FeedPass failed in BoxPS."));
}

void BoxWrapper::BeginFeedPass(int date, boxps::PSAgentBase** agent) const {
  int ret = boxps_ptr_->BeginFeedPass(date, *agent);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginFeedPass failed in BoxPS."));
}

void BoxWrapper::EndFeedPass(boxps::PSAgentBase* agent) const {
  int ret = boxps_ptr_->EndFeedPass(agent);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "EndFeedPass failed in BoxPS."));
}

void BoxWrapper::BeginPass() const {
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginPass failed in BoxPS."));
}

void BoxWrapper::SetTestMode(bool is_test) const {
  boxps_ptr_->SetTestMode(is_test);
}

void BoxWrapper::EndPass(bool need_save_delta) const {
  int ret = boxps_ptr_->EndPass(need_save_delta);
  PADDLE_ENFORCE_EQ(
      ret, 0, platform::errors::PreconditionNotMet("EndPass failed in BoxPS."));
}

void BoxWrapper::PullSparse(const paddle::platform::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size) {
  VLOG(3) << "Begin PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();

  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf =
      memory::AllocShared(place, total_length * sizeof(boxps::FeatureValueGpu));
  boxps::FeatureValueGpu* total_values_gpu =
      reinterpret_cast<boxps::FeatureValueGpu*>(buf->ptr());

  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in PaddleBox now."));
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
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
    VLOG(3) << "Begin call PullSparseGPU in BoxPS";
    pull_boxps_timer.Start();
    int ret =
        boxps_ptr_->PullSparseGPU(total_keys, total_values_gpu,
                                  static_cast<int>(total_length), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Please compile WITH_GPU option, because NCCL doesn't support "
        "windows."));
#endif
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddleBox: PullSparse Only Support CPUPlace or CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which BoxPS costs: " << pull_boxps_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size, const int batch_size) {
  VLOG(3) << "Begin PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_boxps_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length * sizeof(boxps::FeaturePushValueGpu));
  boxps::FeaturePushValueGpu* total_grad_values_gpu =
      reinterpret_cast<boxps::FeaturePushValueGpu*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in PaddleBox now."));
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
    LoDTensor& cached_total_keys_tensor = keys_tensor[device_id];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to boxps struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in BoxPS";
    push_boxps_timer.Start();
    int ret = boxps_ptr_->PushSparseGPU(
        total_keys, total_grad_values_gpu, static_cast<int>(total_length),
        BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId());
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PushSparseGPU failed in BoxPS."));
    push_boxps_timer.Pause();
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Please compile WITH_GPU option, because NCCL doesn't support "
        "windows."));
#endif
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddleBox: PushSparseGrad Only Support CPUPlace or CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which BoxPS cost: " << push_boxps_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}
}  // end namespace framework
}  // end namespace paddle
#endif
