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
#ifdef PADDLE_WITH_BOX_PS
cudaStream_t BoxWrapper::stream_list_[8];
std::shared_ptr<boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;
#endif

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

void BoxWrapper::FeedPass(int date,
                          const std::vector<uint64_t>& feasgin_to_box) const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->FeedPass(date, feasgin_to_box);
  PADDLE_ENFORCE_EQ(ret, 0, "FeedPass failed in BoxPS.");
#endif
}

void BoxWrapper::BeginPass() const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(ret, 0, "BeginPass failed in BoxPS.");
#endif
}

void BoxWrapper::EndPass() const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->EndPass();
  PADDLE_ENFORCE_EQ(ret, 0, "EndPass failed in BoxPS.");
#endif
}

void BoxWrapper::PullSparse(const paddle::platform::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size) {
  PADDLEBOX_LOG << "Begin call PullSparse in BoxWrapper";
  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();
#ifdef PADDLE_WITH_BOX_PS
  if (platform::is_cpu_place(place) || platform::is_gpu_place(place)) {
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    LoDTensor total_keys_tensor;
    int64_t* total_keys =
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place);

    int64_t offset = 0;
    VLOG(3) << "Begin copy keys, key_num[" << keys.size() << "]";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (platform::is_cpu_place(place)) {
        memory::Copy(boost::get<platform::CPUPlace>(place), total_keys + offset,
                     boost::get<platform::CPUPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t));
      } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
        memory::Copy(boost::get<platform::CUDAPlace>(place),
                     total_keys + offset,
                     boost::get<platform::CUDAPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t), nullptr);
#else
        PADDLE_THROW(
            "Please compile WITH_GPU option, and NCCL doesn't support "
            "windows.");
#endif
      }
      offset += slot_lengths[i];
    }
    VLOG(3) << "End copy keys";
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    // Space allocation for FeatureValue is left for boxps
    auto buf = memory::AllocShared(
        place, total_length * sizeof(abacus::FeatureValueGpu));
    abacus::FeatureValueGpu* total_values_gpu =
        reinterpret_cast<abacus::FeatureValueGpu*>(buf->ptr());
    VLOG(3) << "Begin PullSparseGPU";
    pull_boxps_timer.Start();
    if (platform::is_cpu_place(place)) {
      // TODO(hutuxian): should use boxps::FeatureValue in the future
      int ret = boxps_ptr_->PullSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), total_values_gpu,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      int ret = boxps_ptr_->PullSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), total_values_gpu,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseGPU failed in BoxPS.");
      VLOG(3) << "End call boxps_ptr_->PullSparseGPU";
#endif
    }
    pull_boxps_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    if (platform::is_cpu_place(place)) {
      // only support gpu for paddlebox now, and following code don't be tested
      // yet
      offset = 0;
      for (size_t i = 0; i < values.size(); ++i) {
        int64_t fea_num = slot_lengths[i];
        VLOG(3) << "Begin Copy slot[" << i << "] fea_num[" << fea_num << "]";
        for (auto j = 0; j < fea_num; ++j) {
          // Copy the emb from BoxPS to paddle tensor. Since
          // 'show','click','emb'
          // are continuous in memory, so we copy here using the 'show' address
          memory::Copy(
              boost::get<platform::CPUPlace>(place),
              values[i] + j * hidden_size,
              boost::get<platform::CPUPlace>(place),
              reinterpret_cast<float*>(&((total_values_gpu + offset)->show)),
              sizeof(float) * hidden_size);
          ++offset;
        }
      }
    } else {
      this->CopyForPull(place, keys, values, total_values_gpu, slot_lengths,
                        hidden_size, total_length);
    }

    all_timer.Pause();
    PADDLEBOX_LOG << "End PullSparse in BoxWrapper: total cost: "
                  << all_timer.ElapsedSec() << " s, and pull boxps cost: "
                  << pull_boxps_timer.ElapsedSec() << " s";
    VLOG(3) << "End Copy result to tensor";
  } else {
    PADDLE_THROW(
        "PaddleBox: PullSparse Only Support CPUPlace and CUDAPlace Now.");
  }
#endif
  VLOG(3) << "End call PullSparse";
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size, const int batch_size) {
  PADDLEBOX_LOG << "Begin call PushSparseGrad in BoxWrapper";
#ifdef PADDLE_WITH_BOX_PS
  platform::Timer all_timer;
  platform::Timer push_boxps_timer;
  all_timer.Start();
  if (platform::is_cpu_place(place) || platform::is_gpu_place(place)) {
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    LoDTensor total_keys_tensor;
    int64_t* total_keys =
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place);
    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (platform::is_cpu_place(place)) {
        memory::Copy(boost::get<platform::CPUPlace>(place), total_keys + offset,
                     boost::get<platform::CPUPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t));
      } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
        memory::Copy(boost::get<platform::CUDAPlace>(place),
                     total_keys + offset,
                     boost::get<platform::CUDAPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t), nullptr);
#else
        PADDLE_THROW(
            "Please compile WITH_GPU option, and for now NCCL doesn't support "
            "windows.");
#endif
      }
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");
    auto buf = memory::AllocShared(
        place, total_length * sizeof(abacus::FeaturePushValueGpu));
    abacus::FeaturePushValueGpu* total_grad_values_gpu =
        reinterpret_cast<abacus::FeaturePushValueGpu*>(buf->ptr());

    offset = 0;

    if (platform::is_cpu_place(place)) {
      // only support gpu for paddlebox now, and following code don't be tested
      // yet
      for (size_t i = 0; i < grad_values.size(); ++i) {
        int64_t fea_num = slot_lengths[i];
        for (auto j = 0; j < fea_num; ++j) {
          // Copy the emb grad from paddle tensor to BoxPS. Since
          // 'show','click','emb' are continuous in memory, so we copy here
          // using
          // the 'show' address
          memory::Copy(boost::get<platform::CPUPlace>(place),
                       reinterpret_cast<float*>(
                           &((total_grad_values_gpu + offset)->show)),
                       boost::get<platform::CPUPlace>(place),
                       grad_values[i] + j * hidden_size,
                       sizeof(float) * hidden_size);
          ++offset;
        }
      }
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                        hidden_size, total_length, batch_size);
#else
      PADDLE_THROW(
          "Please compile WITH_GPU option, and for now NCCL doesn't support "
          "windows.");
#endif
    }

    push_boxps_timer.Start();
    if (platform::is_cpu_place(place)) {
      int ret = boxps_ptr_->PushSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values_gpu,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      VLOG(3) << "Begin call PushSparseGPU";
      int ret = boxps_ptr_->PushSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values_gpu,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseGPU failed in BoxPS.");
      VLOG(3) << "End call PushSparseGPU";
#endif
    }
    push_boxps_timer.Pause();
    all_timer.Pause();
    PADDLEBOX_LOG << "End PushSparseGrad in BoxWrapper: total cost: "
                  << all_timer.ElapsedSec() << " s, and push boxps cost: "
                  << push_boxps_timer.ElapsedSec() << " s";
  } else {
    PADDLE_THROW(
        "PaddleBox: PushSparse Only Support CPUPlace and CUDAPlace Now.");
  }
  VLOG(3) << "End call PushSparse";
#endif
}
}  // end namespace framework
}  // end namespace paddle
