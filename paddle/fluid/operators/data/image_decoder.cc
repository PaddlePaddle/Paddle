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

#include "paddle/fluid/operators/data/image_decoder.h"

namespace paddle {
namespace operators {
namespace data {

ImageDecoder::ImageDecoder(std::string mode, int dev_id,
                           size_t host_memory_padding,
                           size_t device_memory_padding) 
  : nvjpeg_streams_(2),
    pinned_buffers_(2),
    page_id_(0),
    mode_(mode) {
  platform::SetDeviceId(dev_id);

  // create nvjpeg handle and stream
  PADDLE_ENFORCE_NVJPEG_SUCCESS(
      platform::dynload::nvjpegCreateEx(NVJPEG_BACKEND_HYBRID, &device_allocator_,
                           &pinned_allocator_, 0, &handle_));

  // set pinned/device memory padding
  if (host_memory_padding > 0) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));
  }
  if (device_memory_padding > 0) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
  }

  // create nvjpeg stream
  for (size_t i = 0; i < nvjpeg_streams_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamCreate(handle_, &nvjpeg_streams_[i]));
  }

  // create decode params, decoder and state
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsCreate(handle_, &decode_params_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HYBRID, &decoder_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderStateCreate(handle_, decoder_, &state_));

  // create device & pinned buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferDeviceCreate(handle_, &device_allocator_, &device_buffer_));
  for (size_t i = 0; i < pinned_buffers_.size(); i++) {
    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferPinnedCreate(handle_, &pinned_allocator_, &pinned_buffers_[i]));
  }
}

ImageDecoder::~ImageDecoder() {
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

  // destroy nvjpeg handle at last
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDestroy(handle_));
}

void ImageDecoder::CPUDecodeRandomCrop(
                        const uint8_t* data, size_t length,
                        RandomROIGenerator* roi_generator,
                        unsigned char* workspace, size_t workspace_size,
                        framework::LoDTensor* out, platform::Place place) {
  VLOG(4) << "CPUDecodeRandomCropResize enter";
#ifdef PADDLE_WITH_OPENCV
  cv::Mat image =
      cv::imdecode(cv::Mat(1, length, CV_8UC1, const_cast<unsigned char*>(data)), cv::IMREAD_COLOR);
  
  auto* image_data = image.data;
  int height = image.rows;
  int width = image.cols;
  if (roi_generator) {
    ROI roi;
    roi_generator->GenerateRandomROI(image.cols, image.rows, &roi);
    cv::Rect cv_roi;
    cv_roi.x = roi.x;
    cv_roi.y = roi.y;
    cv_roi.width = roi.w;
    cv_roi.height = roi.h;
    height = roi.h;
    width = roi.w;

    // todo jianglielin: why not work?
    // cropped.data = data;
    cv::Mat cropped;
    image(cv_roi).copyTo(cropped);
    image_data = cropped.data;
  }

  framework::LoDTensor cpu_tensor;
  std::vector<int64_t> out_shape = {height, width, 3};
  cpu_tensor.Resize(framework::make_ddim(out_shape));
  // allocate memory and assign to out_image
  auto* cpu_data = cpu_tensor.mutable_data<uint8_t>(platform::CPUPlace());

  std::memcpy(cpu_data, image_data, 3 * height * width);
  TensorCopySync(cpu_tensor, place, out);
#else
  PADDLE_THROW(platform::errors::Fatal(
      "Nvjpeg decode failed and Paddle is not compiled with OpenCV"));
#endif
}

nvjpegStatus_t ImageDecoder::ParseDecodeParams(
    const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out,
    RandomROIGenerator* roi_generator, nvjpegImage_t* out_image,
    platform::Place place) {
  int components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  
  nvjpegStatus_t status = platform::dynload::nvjpegGetImageInfo(handle_, bit_stream, bit_len,
                         &components, &subsampling, widths, heights);

  if (status != NVJPEG_STATUS_SUCCESS) return status;

  int64_t width = static_cast<int64_t>(widths[0]);
  int64_t height = static_cast<int64_t>(heights[0]);

  nvjpegOutputFormat_t output_format;
  int output_components;

  if (mode_ == "unchanged") {
    if (components == 1) {
      output_format = NVJPEG_OUTPUT_Y;
      output_components = 1;
    } else if (components == 3) {
      output_format = NVJPEG_OUTPUT_RGBI;
      output_components = 3;
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "The provided mode(%s) does not support components(%d)",
          mode_, components));
    }
  } else if (mode_ == "gray") {
    output_format = NVJPEG_OUTPUT_Y;
    output_components = 1;
  } else if (mode_ == "rgb") {
    output_format = NVJPEG_OUTPUT_RGBI;
    output_components = 3;
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The provided mode is not supported for JPEG files on GPU: %s!", mode_));
  }

  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format));

  if (roi_generator) {
    ROI roi;
    roi_generator->GenerateRandomROI(width, height, &roi);

    PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsSetROI(decode_params_, roi.x, roi.y, roi.w, roi.h));
    height = roi.h;
    width = roi.w;
  }

  std::vector<int64_t> out_shape = {height, width, output_components};
  out->Resize(framework::make_ddim(out_shape));

  // allocate memory and assign to out_image
  auto* data = out->mutable_data<uint8_t>(place);
  out_image->channel[0] = data;
  out_image->pitch[0] = output_components * width;

  return NVJPEG_STATUS_SUCCESS;
}

nvjpegStatus_t ImageDecoder::GPUDecodeRandomCrop(const uint8_t* bit_stream, size_t bit_len, nvjpegImage_t* out_image) {
  auto buffer = pinned_buffers_[page_id_];
  auto stream = nvjpeg_streams_[page_id_];
  page_id_ ^= 1;

  // decode jpeg in host to pinned buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamParse(handle_, bit_stream, bit_len, false, false, stream));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachPinnedBuffer(state_, buffer));
  nvjpegStatus_t status = platform::dynload::nvjpegDecodeJpegHost(handle_, decoder_, state_, decode_params_, stream);
  if (status != NVJPEG_STATUS_SUCCESS) return status;

  // transfer and decode to device buffer
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachDeviceBuffer(state_, device_buffer_));
  PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_, stream, cuda_stream_));
  status = platform::dynload::nvjpegDecodeJpegDevice(handle_, decoder_, state_, out_image, nullptr);
  return status;
}

void ImageDecoder::Run(
    const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out,
    RandomROIGenerator* roi_generator, platform::Place& place) {
  nvjpegImage_t image;
  nvjpegStatus_t status = ParseDecodeParams(bit_stream, bit_len, out, roi_generator, &image, place);
  if (status != NVJPEG_STATUS_SUCCESS) {
    CPUDecodeRandomCrop(bit_stream, bit_len, roi_generator, nullptr, 0, out, place);
    return;
  }
  status = GPUDecodeRandomCrop(bit_stream, bit_len, &image);
  if (status != NVJPEG_STATUS_SUCCESS) {
    CPUDecodeRandomCrop(bit_stream, bit_len, roi_generator, nullptr, 0, out, place);
  }
}

ImageDecoderThreadPool::ImageDecoderThreadPool(
    const int num_threads, const std::string mode, const int dev_id,
    const size_t host_memory_padding, const size_t device_memory_padding)
  : threads_(num_threads),
    mode_(mode),
    dev_id_(dev_id),
    shutdown_(false),
    running_(false),
    completed_(false),
    outstand_tasks_(0) {
  PADDLE_ENFORCE_GT(num_threads, 0, platform::errors::InvalidArgument(
                    "num_threads shoule be a positive interger, "
                    "but got %d", num_threads));
  for (int i = 0; i < num_threads; i++) {
    threads_.emplace_back(
        std::thread(std::bind(&ImageDecoderThreadPool::ThreadLoop,
            this, i, host_memory_padding, device_memory_padding)));
  }
}

ImageDecoderThreadPool::~ImageDecoderThreadPool() { ShutDown(); }

void ImageDecoderThreadPool::AddTask(std::shared_ptr<ImageDecodeTask> task) {
  task_queue_.push_back(task);
}

void ImageDecoderThreadPool::RunAll(const bool wait, const bool sort) {
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

void ImageDecoderThreadPool::WaitTillTasksCompleted() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_cond_.wait(lock, [this] { return this->completed_; });
  running_ = false;
}

void ImageDecoderThreadPool::ShutDown() {
  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  shutdown_ = true;
  running_cond_.notify_all();
  lock.unlock();

  task_queue_.clear();

  for (auto& thread : threads_) {
    if (thread.joinable())  thread.join();
  }
}

void ImageDecoderThreadPool::SortTaskByLengthDescend() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::sort(task_queue_.begin(), task_queue_.end(),
      [](const std::shared_ptr<ImageDecodeTask> a,
         const std::shared_ptr<ImageDecodeTask> b) {
          return b->bit_len < a->bit_len;
      });
}

void ImageDecoderThreadPool::ThreadLoop(
      const int thread_idx, const size_t host_memory_padding,
      const size_t device_memory_padding) {
  ImageDecoder* decoder = new ImageDecoder(mode_, dev_id_,
                                           host_memory_padding,
                                           device_memory_padding);

  while (!shutdown_) {
    std::unique_lock<std::mutex> lock(mutex_);
    running_cond_.wait(lock, [this] { return (running_ && !task_queue_.empty()) || shutdown_; });
    if (shutdown_) break;

    auto task = task_queue_.front();
    task_queue_.pop_front();
    outstand_tasks_++;
    lock.unlock();

    decoder->Run(task->bit_stream, task->bit_len, task->tensor,
                 task->roi_generator, task->place);

    lock.lock();
    outstand_tasks_--;
    if (outstand_tasks_ == 0 && task_queue_.empty()) {
      completed_ = true;
      lock.unlock();
      completed_cond_.notify_one();
    }
  }
}

// initialization static variables out of ImageDecoderThreadPoolManager
ImageDecoderThreadPoolManager* ImageDecoderThreadPoolManager::pm_instance_ptr_ = nullptr;
std::mutex ImageDecoderThreadPoolManager::m_;

}  // namespace data
}  // namespace operators
}  // namespace paddle
