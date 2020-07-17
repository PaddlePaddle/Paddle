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

#pragma once
#ifdef PADDLE_WITH_BOX_PS
#include <vector>
namespace paddle {
namespace framework {

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
void BoxWrapper::PullSparseCase(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<float*>& values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim) {
  VLOG(3) << "Begin PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();

  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length *
                 sizeof(boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>));
  boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* total_values_gpu =
      reinterpret_cast<boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*>(
          buf->ptr());

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
    int ret = boxps_ptr_->PullSparseGPU(
        total_keys, reinterpret_cast<void*>(total_values_gpu),
        static_cast<int>(total_length), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values,
                      reinterpret_cast<void*>(total_values_gpu), gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      expand_embed_dim, total_length);
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

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
void BoxWrapper::PushSparseGradCase(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size) {
  VLOG(3) << "Begin PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_boxps_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place,
      total_length *
          sizeof(boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>));
  boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*
      total_grad_values_gpu = reinterpret_cast<
          boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*>(
          buf->ptr());
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
                      hidden_size, expand_embed_dim, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in BoxPS";
    push_boxps_timer.Start();
    int ret = boxps_ptr_->PushSparseGPU(
        total_keys, reinterpret_cast<void*>(total_grad_values_gpu),
        static_cast<int>(total_length),
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

}  // namespace framework
}  // namespace paddle
#endif
