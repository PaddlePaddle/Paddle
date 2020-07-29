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
//  VLOG(3) << "Begin PullSparse";
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_pull_timer;
  platform::Timer& pull_boxps_timer = dev.boxps_pull_timer;
#else
  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
#endif
  all_timer.Resume();

  // construct slot_level lod info
  auto slot_lengths_lod = slot_lengths;
  int slot_num = static_cast<int>(slot_lengths.size());
  for (int i = 1; i < slot_num; i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  int64_t total_length = slot_lengths_lod[slot_num - 1];
  size_t total_bytes = reinterpret_cast<size_t>(
      total_length *
      sizeof(boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>));
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  dev.total_key_length = total_length;
  auto& pull_buf = dev.pull_push_buf;
  if (pull_buf == nullptr) {
    pull_buf = memory::AllocShared(place, total_bytes);
  } else if (total_bytes > pull_buf->size()) {
    auto buf = memory::AllocShared(place, total_bytes);
    pull_buf.swap(buf);
    buf = nullptr;
  }
#else
  auto pull_buf = memory::AllocShared(place, total_bytes);
#endif
  boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* total_values_gpu =
      reinterpret_cast<boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*>(
          pull_buf->ptr());

  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in PaddleBox now."));
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(
                          BOOST_GET_CONST(platform::CUDAPlace, place)))
                      ->stream();
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        dev.keys_tensor.mutable_data<int64_t>({total_length, 1}, place));
    int* total_dims = reinterpret_cast<int*>(
        dev.dims_tensor.mutable_data<int>({total_length, 1}, place));
    if (dev.gpu_keys_ptr == nullptr) {
      dev.gpu_keys_ptr =
          memory::AllocShared(place, keys.size() * sizeof(uint64_t*));
    }

    int* key2slot = reinterpret_cast<int*>(
        dev.keys2slot.mutable_data<int>({total_length, 1}, place));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(dev.gpu_keys_ptr->ptr());
    int64_t* slot_lens = reinterpret_cast<int64_t*>(
        dev.slot_lens.mutable_data<int64_t>({slot_num, 1}, place));
    cudaMemcpyAsync(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(slot_lens, slot_lengths_lod.data(),
                    slot_lengths.size() * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    this->CopyKeys(place, gpu_keys, total_keys, slot_lens, slot_num,
                   static_cast<int>(total_length), key2slot);

    pull_boxps_timer.Resume();
    int ret = boxps_ptr_->PullSparseGPU(
        total_keys, reinterpret_cast<void*>(total_values_gpu),
        static_cast<int>(total_length), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();

    if (dev.gpu_values_ptr == nullptr) {
      dev.gpu_values_ptr =
          memory::AllocShared(place, values.size() * sizeof(float*));
    }
    float** gpu_values = reinterpret_cast<float**>(dev.gpu_values_ptr->ptr());
    cudaMemcpyAsync(gpu_values, values.data(), values.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    this->CopyForPull(place, gpu_keys, gpu_values, total_values_gpu, slot_lens,
                      slot_num, key2slot, hidden_size, expand_embed_dim,
                      total_length, total_dims);
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
}

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
void BoxWrapper::PushSparseGradCase(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;
#else
  platform::Timer all_timer;
  platform::Timer push_boxps_timer;
#endif
  all_timer.Resume();

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  int64_t total_length = dev.total_key_length;
  // std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  size_t total_bytes = reinterpret_cast<size_t>(
      total_length *
      sizeof(boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>));
  auto& push_buf = dev.pull_push_buf;
  if (push_buf == nullptr) {
    push_buf = memory::AllocShared(place, total_bytes);
  } else if (total_bytes > push_buf->size()) {
    auto buf = memory::AllocShared(place, total_bytes);
    push_buf.swap(buf);
    buf = nullptr;
  }
#else
  auto push_buf = memory::AllocShared(place, total_bytes);
#endif
  boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*
      total_grad_values_gpu = reinterpret_cast<
          boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>*>(
          push_buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in PaddleBox now."));
  } else if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(
                          BOOST_GET_CONST(platform::CUDAPlace, place)))
                      ->stream();
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
    int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
    int slot_num = static_cast<int>(slot_lengths.size());
    if (!dev.d_slot_vector.IsInitialized()) {
      int* buf_slot_vector = reinterpret_cast<int*>(
          dev.d_slot_vector.mutable_data<int>({slot_num, 1}, place));
      cudaMemcpyAsync(buf_slot_vector, slot_vector_.data(),
                      slot_num * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    const int64_t* slot_lens =
        reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
    const int* d_slot_vector = dev.d_slot_vector.data<int>();
    const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());
    float** gpu_values = reinterpret_cast<float**>(dev.gpu_values_ptr->ptr());
    cudaMemcpyAsync(gpu_values, grad_values.data(),
                    grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice,
                    stream);

    this->CopyForPush(place, gpu_values, total_grad_values_gpu, d_slot_vector,
                      slot_lens, slot_num, hidden_size, expand_embed_dim,
                      total_length, batch_size, total_dims, key2slot);

    push_boxps_timer.Resume();
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
}

}  // namespace framework
}  // namespace paddle
#endif
