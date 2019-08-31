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
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
#ifdef PADDLE_WITH_BOX_PS
std::shared_ptr<paddle::boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;
#endif

int BoxWrapper::GetDate() const {
  time_t now = time(0);
  tm t;
#ifdef _WIN32
  localtime_s(&t, &now);
#else
  localtime_r(&now, &t);
#endif
  char buf[10];
  snprintf(buf, sizeof(buf), "%04d%02d%02d", (1900 + t.tm_year), (1 + t.tm_mon),
           t.tm_mday);
  return atoi(buf);
}

void BoxWrapper::FeedPass(const std::vector<uint64_t>& feasgin_to_box) const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->FeedPass(GetDate(), feasgin_to_box);
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
#ifdef PADDLE_WITH_BOX_PS
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
            "Please compile WITH_GPU option, and NCCL doesn't support "
            "windows.");
#endif
      }
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    // Space allocation for FeatureValue is left for boxps
    paddle::boxps::FeatureValue* total_values;
    if (platform::is_cpu_place(place)) {
      int ret = boxps_ptr_->PullSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), &total_values,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      int ret = boxps_ptr_->PullSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), &total_values,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseGPU failed in BoxPS.");
#endif
    }

    offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        // Copy the emb from BoxPS to paddle tensor. Since 'show','click','emb'
        // are continuous in memory, so we copy here using the 'show' address
        if (platform::is_cpu_place(place)) {
          memory::Copy(
              boost::get<platform::CPUPlace>(place),
              values[i] + j * hidden_size,
              boost::get<platform::CPUPlace>(place),
              reinterpret_cast<float*>(&((total_values + offset)->show)),
              sizeof(float) * hidden_size);
        } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
          memory::Copy(
              boost::get<platform::CUDAPlace>(place),
              values[i] + j * hidden_size,
              boost::get<platform::CUDAPlace>(place),
              reinterpret_cast<float*>(&((total_values + offset)->show)),
              sizeof(float) * hidden_size, nullptr);
#endif
        }
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total emb values length should "
                      "be equal to the sum of length of all input tensors.");

  } else {
    PADDLE_THROW(
        "PaddleBox: PullSparse Only Support CPUPlace and CUDAPlace Now.");
  }
#endif
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size) {
#ifdef PADDLE_WITH_BOX_PS
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
        place, total_length * sizeof(paddle::boxps::FeaturePushValue));
    paddle::boxps::FeaturePushValue* total_grad_values =
        reinterpret_cast<paddle::boxps::FeaturePushValue*>(buf->ptr());
    offset = 0;
    for (size_t i = 0; i < grad_values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        // Copy the emb grad from paddle tensor to BoxPS. Since
        // 'show','click','emb' are continuous in memory, so we copy here using
        // the 'show' address
        if (platform::is_cpu_place(place)) {
          memory::Copy(
              boost::get<platform::CPUPlace>(place),
              reinterpret_cast<float*>(&((total_grad_values + offset)->show)),
              boost::get<platform::CPUPlace>(place),
              grad_values[i] + j * hidden_size, sizeof(float) * hidden_size);
        } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
          memory::Copy(
              boost::get<platform::CUDAPlace>(place),
              reinterpret_cast<float*>(&((total_grad_values + offset)->show)),
              boost::get<platform::CUDAPlace>(place),
              grad_values[i] + j * hidden_size, sizeof(float) * hidden_size,
              nullptr);
#endif
        }
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total emb grad values "
                      "length should be equal to the sum of length of all "
                      "input tensors.");
    if (platform::is_cpu_place(place)) {
      int ret = boxps_ptr_->PushSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      int ret = boxps_ptr_->PushSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseGPU failed in BoxPS.");
#endif
    }
  } else {
    PADDLE_THROW(
        "PaddleBox: PushSparse Only Support CPUPlace and CUDAPlace Now.");
  }
#endif
}
}  // end namespace framework
}  // end namespace paddle
