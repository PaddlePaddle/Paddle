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

#include "paddle/fluid/operators/data/nvjpeg_decoder.h"

namespace paddle {
namespace operators {
namespace data {

NvjpegDecoder::NvjpegDecoder(std::string mode) 
  : nvjpeg_streams_(2),
    pinned_buffers_(2),
    page_id_(0),
    mode_(mode) {
  // create cuda stream
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));

  // create nvjpeg handle and stream
  // device_allocator_.dev_malloc = &cudaMalloc;
  // device_allocator_.dev_free = &cudaFree;
  // pinned_allocator_.pinned_malloc = &cudaMallocHost;
  // pinned_allocator_.pinned_free = &cudaFreeHost;
  PADDLE_ENFORCE_NVJPEG_SUCCESS(
      platform::dynload::nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &device_allocator_,
                           &pinned_allocator_, 0, &handle_));
  for (size_t i = 0; i < nvjpeg_streams_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamCreate(handle_, &nvjpeg_streams_[i]));
  }

  // create decode params, decoder and state
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsCreate(handle_, &decode_params_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_DEFAULT, &decoder_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderStateCreate(handle_, decoder_, &state_));

  // create device & pinned buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferDeviceCreate(handle_, &device_allocator_, &device_buffer_));
  for (size_t i = 0; i < pinned_buffers_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferPinnedCreate(handle_, &pinned_allocator_, &pinned_buffers_[i]));
  }
}

NvjpegDecoder::~NvjpegDecoder() {
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(cuda_stream_));

  // destroy nvjpeg streams
  for (size_t i = 0; i < nvjpeg_streams_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamDestroy(nvjpeg_streams_[i]));
  }

  // destroy decode params, decoder and state
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsDestroy(decode_params_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderDestroy(decoder_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStateDestroy(state_));

  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferDeviceDestroy(device_buffer_));
  for (size_t i = 0; i < pinned_buffers_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferPinnedDestroy(pinned_buffers_[i]));
  }

  // destroy nvjpeg handle and cuda stream at last
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDestroy(handle_));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(cuda_stream_));
}

void NvjpegDecoder::ParseOutputInfo(
    const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out,
    nvjpegImage_t* out_image, platform::Place place) {
  int components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  PADDLE_ENFORCE_NVJPEG_SUCCESS(
      platform::dynload::nvjpegGetImageInfo(handle_, bit_stream, bit_len,
                         &components, &subsampling, widths, heights));

  int width = widths[0];
  int height = heights[0];

  nvjpegOutputFormat_t output_format;
  int output_components;

  if (mode_ == "unchanged") {
    if (components == 1) {
      output_format = NVJPEG_OUTPUT_Y;
      output_components = 1;
    } else if (components == 3) {
      output_format = NVJPEG_OUTPUT_RGB;
      output_components = 3;
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "The provided mode is not supported for JPEG files on GPU"));
    }
  } else if (mode_ == "gray") {
    output_format = NVJPEG_OUTPUT_Y;
    output_components = 1;
  } else if (mode_ == "rgb") {
    output_format = NVJPEG_OUTPUT_RGB;
    output_components = 3;
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The provided mode is not supported for JPEG files on GPU"));
  }

  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format));

  std::vector<int64_t> out_shape = {output_components, height, width};
  out->Resize(framework::make_ddim(out_shape));

  // allocate memory and assign to out_image
  auto* data = out->mutable_data<uint8_t>(place);
  for (int c = 0; c < output_components; c++) {
    out_image->channel[c] = data + c * width * height;
    out_image->pitch[c] = width;
  }
}

void NvjpegDecoder::Decode(const uint8_t* bit_stream, size_t bit_len, nvjpegImage_t* out_image) {
  auto buffer = pinned_buffers_[page_id_];
  auto stream = nvjpeg_streams_[page_id_];
  page_id_ ^= 1;

  // decode jpeg in host to pinned buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachPinnedBuffer(state_, buffer));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamParse(handle_, bit_stream, bit_len, false, false, stream));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegHost(handle_, decoder_, state_, decode_params_, stream));

  // transfer and decode to device buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachDeviceBuffer(state_, device_buffer_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_, stream, cuda_stream_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegDevice(handle_, decoder_, state_, out_image, cuda_stream_));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(cuda_stream_));
}

void NvjpegDecoder::Run(const uint8_t* bit_stream, size_t bit_len,
                        framework::LoDTensor* out, platform::Place& place) {
  nvjpegImage_t image;
  ParseOutputInfo(bit_stream, bit_len, out, &image, place);
  Decode(bit_stream, bit_len, &image);
}

NvjpegDecoderThreadPool::NvjpegDecoderThreadPool(const int num_threads, const std::string mode)
  : threads_(num_threads),
    mode_(mode),
    shutdown_(false),
    running_(false),
    completed_(false),
    outstand_tasks_(0) {
  PADDLE_ENFORCE_GT(num_threads, 0, platform::errors::InvalidArgument(
                    "num_threads shoule be a positive interger, "
                    "but got %d", num_threads));
  for (int i = 0; i < num_threads; i++) {
    threads_.emplace_back(
        std::thread(std::bind(&NvjpegDecoderThreadPool::ThreadLoop, this, i)));
  }
}

NvjpegDecoderThreadPool::~NvjpegDecoderThreadPool() { Shutdown(); }

void NvjpegDecoderThreadPool::AddTask(std::shared_ptr<NvjpegDecodeTask> task) {
  task_queue_.push_back(task);
}

void NvjpegDecoderThreadPool::RunAll(const bool wait, const bool sort) {
  // Sort images in length desending order
  if (sort) SortTaskByLengthDescend();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_ = false;
    running_ = true;
  }
  running_cond_.notify_all();

  if (wait) WaitTillTasksCompleted();
}

void NvjpegDecoderThreadPool::WaitTillTasksCompleted() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_cond_.wait(lock, [this] { return this->completed_; });
  running_ = false;
}

void NvjpegDecoderThreadPool::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);

  running_ = false;
  shutdown_.store(true);
  running_cond_.notify_all();

  task_queue_.clear();

  for (auto &thread : threads_) {
    thread.join();
  }
}

void NvjpegDecoderThreadPool::SortTaskByLengthDescend() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::sort(task_queue_.begin(), task_queue_.end(),
      [](const std::shared_ptr<NvjpegDecodeTask> a,
         const std::shared_ptr<NvjpegDecodeTask> b) {
          return b->bit_len < a->bit_len;
      });
}

void NvjpegDecoderThreadPool::ThreadLoop(const int thread_idx) {
  NvjpegDecoder* decoder = new NvjpegDecoder(mode_);

  while (!shutdown_.load()) {
    std::unique_lock<std::mutex> lock(mutex_);
    running_cond_.wait(lock, [this] { return running_ && !task_queue_.empty(); });
    if (shutdown_.load()) break;

    auto task = task_queue_.front();
    task_queue_.pop_front();
    outstand_tasks_++;
    lock.unlock();

    decoder->Run(task->bit_stream, task->bit_len, task->tensor, task->place);

    lock.lock();
    outstand_tasks_--;
    if (outstand_tasks_ == 0 && task_queue_.empty()) {
      completed_ = true;
      lock.unlock();
      completed_cond_.notify_one();
    }
  }
}

}  // namespace data
}  // namespace operators
}  // namespace paddle
