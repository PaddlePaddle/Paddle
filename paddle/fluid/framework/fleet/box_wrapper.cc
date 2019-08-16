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
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
std::shared_ptr<paddle::boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;

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

int BoxWrapper::FeedPass(const std::vector<uint64_t>& feasgin_to_box) const {
  boxps_ptr_->FeedPass(GetDate(), feasgin_to_box);
  return 0;
}

int BoxWrapper::BeginPass() const {
  boxps_ptr_->BeginPass();
  return 0;
}

int BoxWrapper::EndPass() const {
  boxps_ptr_->EndPass();
  return 0;
}

int BoxWrapper::PullSparse(const paddle::platform::Place& place,
                           const std::vector<const uint64_t*>& keys,
                           const std::vector<float*>& values,
                           const std::vector<int64_t>& slot_lengths,
                           const int hidden_size) {
  if (platform::is_cpu_place(place)) {
    VLOG(10) << "PaddleBox: PullSparse in CPUPlace";
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    uint64_t* total_keys = new uint64_t[total_length];
    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      memcpy(total_keys + offset, keys[i], slot_lengths[i] * sizeof(uint64_t));
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    // Space allocation for FeatureValue is left for boxps
    paddle::boxps::FeatureValue* total_values;
    boxps_ptr_->PullSparseCPU(total_keys, &total_values,
                              static_cast<int>(total_length));

    offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        memcpy(values[i] + j * hidden_size,
               reinterpret_cast<float*>(&((total_values + offset)->show)),
               sizeof(float) * hidden_size);
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total emb values length should "
                      "be equal to the sum of length of all input tensors.");

    // All of these vectors should be free by boxps, right?
    delete[] total_keys;
    delete[] total_values;  // should be free in boxps, but for stub free here
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    VLOG(10) << "PaddleBox: PullSparse in CUDAPlace";
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place).GetDeviceId());

    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);

    // TODO(hutuxian): should we use system_allocator or mutable_data() instead
    // of calling cudaMalloc directly?
    uint64_t* total_keys;
    cudaError_t result =
        cudaMalloc(&total_keys, total_length * sizeof(uint64_t));
    PADDLE_ENFORCE_EQ(result, cudaSuccess, "PaddleBox: cudaMalloc failed.");

    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      cudaMemcpy(total_keys + offset, keys[i],
                 slot_lengths[i] * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    // Space allocation for FeatureValue is left for boxps
    paddle::boxps::FeatureValue* total_values;
    boxps_ptr_->PullSparseGPU(
        total_keys, &total_values, static_cast<int>(total_length),
        boost::get<platform::CUDAPlace>(place).GetDeviceId());

    offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        cudaMemcpy(values[i] + j * hidden_size,
                   reinterpret_cast<float*>(&((total_values + offset)->show)),
                   sizeof(float) * hidden_size, cudaMemcpyDeviceToDevice);
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total emb values length should "
                      "be equal to the sum of length of all input tensors.");

    // All of these vectors should be free by boxps, right?
    cudaFree(total_keys);
    cudaFree(total_values);  // should be free in boxps, but for stub free here
#else
    PADDLE_THROW(
        "Please compile WITH_GPU option, and for now NCCL doesn't support "
        "windows.");
#endif
  } else {
    VLOG(3)
        << "PaddleBox: PullSparse Only support CPUPlace and CUDAPlace now.\n";
    return 1;
  }
  return 0;
}

int BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                               const std::vector<const uint64_t*>& keys,
                               const std::vector<const float*>& grad_values,
                               const std::vector<int64_t>& slot_lengths,
                               const int hidden_size) {
  if (platform::is_cpu_place(place)) {
    VLOG(10) << "PaddleBox: PushSparse in CPUPlace";
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    uint64_t* total_keys = new uint64_t[total_length];
    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      memcpy(total_keys + offset, keys[i], slot_lengths[i] * sizeof(uint64_t));
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    paddle::boxps::FeaturePushValue* total_grad_values =
        new paddle::boxps::FeaturePushValue[total_length];

    offset = 0;
    for (size_t i = 0; i < grad_values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        memcpy(reinterpret_cast<float*>(&((total_grad_values + offset)->show)),
               grad_values[i] + j * hidden_size, sizeof(float) * hidden_size);
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total emb grad values "
                      "length should be equal to the sum of length of all "
                      "input tensors.");

    boxps_ptr_->PushSparseCPU(total_keys, total_grad_values,
                              static_cast<int>(total_length));
    delete[] total_keys;
    delete[] total_grad_values;
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    VLOG(10) << "PaddleBox: PushSparse in CUDAPlace";
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place).GetDeviceId());

    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);

    uint64_t* total_keys;
    cudaError_t result =
        cudaMalloc(&total_keys, total_length * sizeof(uint64_t));
    PADDLE_ENFORCE_EQ(result, cudaSuccess, "PaddleBox: cudaMalloc failed.");

    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      cudaMemcpy(total_keys + offset, keys[i],
                 slot_lengths[i] * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    paddle::boxps::FeaturePushValue* total_grad_values;
    result = cudaMalloc(&total_grad_values,
                        total_length * sizeof(paddle::boxps::FeaturePushValue));
    PADDLE_ENFORCE_EQ(result, cudaSuccess, "PaddleBox: cudaMalloc failed.");

    offset = 0;
    for (size_t i = 0; i < grad_values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        cudaMemcpy(
            reinterpret_cast<float*>(&((total_grad_values + offset)->show)),
            grad_values[i] + j * hidden_size, sizeof(float) * hidden_size,
            cudaMemcpyDeviceToDevice);
        ++offset;
      }
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total emb grad values "
                      "length should be equal to the sum of length of all "
                      "input tensors.");

    boxps_ptr_->PushSparseGPU(
        total_keys, total_grad_values, static_cast<int>(total_length),
        boost::get<platform::CUDAPlace>(place).GetDeviceId());
    cudaFree(total_keys);
    cudaFree(total_grad_values);
#else
    PADDLE_THROW(
        "Please compile WITH_GPU option, and for now NCCL doesn't support "
        "windows.");
#endif
  } else {
    VLOG(3)
        << "PaddleBox: PullSparse Only support CPUPlace and CUDAPlace now.\n";
    return 1;
  }
  return 0;
}
}  // end namespace framework
}  // end namespace paddle
