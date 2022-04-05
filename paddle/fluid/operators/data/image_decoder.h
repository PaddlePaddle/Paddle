/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <vector>

#ifdef PADDLE_WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

#include "paddle/fluid/operators/data/random_roi_generator.h"

namespace paddle {
namespace operators {
namespace data {

static int dev_malloc(void** p, size_t s) {
  return static_cast<int>(cudaMalloc(p, s));
}

static int dev_free(void* p) { return static_cast<int>(cudaFree(p)); }

static int host_malloc(void** p, size_t s, unsigned int f) {
  return static_cast<int>(cudaHostAlloc(p, s, f));
}

static int host_free(void* p) { return static_cast<int>(cudaFreeHost(p)); }

struct ImageDecodeTask {
  const uint8_t* bit_stream;
  size_t bit_len;
  framework::LoDTensor* tensor;
  RandomROIGenerator* roi_generator;
  platform::Place place;
};

class ImageDecoder {
 public:
  ImageDecoder(int dev_id, size_t host_memory_padding = 0,
               size_t device_memory_padding = 0);

  ~ImageDecoder();

  void Run(const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out,
           RandomROIGenerator* roi_generator, const platform::Place& place);

 private:
  DISABLE_COPY_AND_ASSIGN(ImageDecoder);

  void CPUDecodeRandomCrop(const uint8_t* data, size_t length,
                           RandomROIGenerator* roi_generator,
                           unsigned char* workspace, size_t workspace_size,
                           framework::LoDTensor* out, platform::Place place);

  nvjpegStatus_t ParseDecodeParams(const uint8_t* bit_stream, size_t bit_len,
                                   framework::LoDTensor* out,
                                   RandomROIGenerator* roi_generator,
                                   nvjpegImage_t* out_image,
                                   platform::Place place);

  nvjpegStatus_t GPUDecodeRandomCrop(const uint8_t* bit_stream, size_t bit_len,
                                     nvjpegImage_t* out_image);

  cudaStream_t cuda_stream_ = nullptr;
  std::vector<nvjpegJpegStream_t> nvjpeg_streams_;

  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  nvjpegJpegDecoder_t decoder_ = nullptr;
  nvjpegDecodeParams_t decode_params_ = nullptr;

  nvjpegPinnedAllocator_t pinned_allocator_ = {&host_malloc, &host_free};
  nvjpegDevAllocator_t device_allocator_ = {&dev_malloc, &dev_free};
  std::vector<nvjpegBufferPinned_t> pinned_buffers_;
  nvjpegBufferDevice_t device_buffer_ = nullptr;

  int page_id_;
};

class ImageDecoderThreadPool {
 public:
  ImageDecoderThreadPool(const int num_threads, const int dev_id,
                         size_t host_memory_padding,
                         size_t device_memory_padding);

  ~ImageDecoderThreadPool();

  void AddTask(std::shared_ptr<ImageDecodeTask> task);

  void RunAll(const bool wait, const bool sort = true);

  void WaitTillTasksCompleted();

  void ShutDown();

 private:
  DISABLE_COPY_AND_ASSIGN(ImageDecoderThreadPool);

  void SortTaskByLengthDescend();

  void ThreadLoop(const int thread_idx, const size_t host_memory_padding,
                  const size_t device_memory_padding);

  std::vector<std::thread> threads_;
  int dev_id_;

  std::deque<std::shared_ptr<ImageDecodeTask>> task_queue_;
  std::mutex mutex_;

  bool shutdown_;
  std::condition_variable running_cond_;
  bool running_;
  std::condition_variable completed_cond_;
  bool completed_;

  int outstand_tasks_;
};

class ImageDecoderThreadPoolManager {
 private:
  DISABLE_COPY_AND_ASSIGN(ImageDecoderThreadPoolManager);

  static ImageDecoderThreadPoolManager* pm_instance_ptr_;
  static std::mutex m_;

  std::map<int64_t, std::unique_ptr<ImageDecoderThreadPool>> prog_id_to_pool_;

 public:
  static ImageDecoderThreadPoolManager* Instance() {
    if (pm_instance_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lk(m_);
      if (pm_instance_ptr_ == nullptr) {
        pm_instance_ptr_ = new ImageDecoderThreadPoolManager;
      }
    }
    return pm_instance_ptr_;
  }

  ImageDecoderThreadPool* GetDecoderThreadPool(
      const int64_t program_id, const int num_threads, const int dev_id,
      const size_t host_memory_padding, const size_t device_memory_padding) {
    auto iter = prog_id_to_pool_.find(program_id);
    if (iter == prog_id_to_pool_.end()) {
      prog_id_to_pool_[program_id] =
          std::unique_ptr<ImageDecoderThreadPool>(new ImageDecoderThreadPool(
              num_threads, dev_id, host_memory_padding, device_memory_padding));
    }
    return prog_id_to_pool_[program_id].get();
  }

  void ShutDownDecoder(const int64_t program_id) {
    auto iter = prog_id_to_pool_.find(program_id);
    if (iter != prog_id_to_pool_.end()) {
      iter->second.get()->ShutDown();
      prog_id_to_pool_.erase(program_id);
    }
  }

  void ShutDown() {
    if (prog_id_to_pool_.empty()) return;

    std::lock_guard<std::mutex> lk(m_);
    auto iter = prog_id_to_pool_.begin();
    for (; iter != prog_id_to_pool_.end(); iter++) {
      if (iter->second.get()) iter->second.get()->ShutDown();
    }
  }

  ImageDecoderThreadPoolManager() {}

  ~ImageDecoderThreadPoolManager() { ShutDown(); }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
