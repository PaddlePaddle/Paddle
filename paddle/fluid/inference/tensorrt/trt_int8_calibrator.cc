// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const TRT_NOEXCEPT { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator(
    const std::unordered_map<std::string, size_t>& buffers, int batch_size,
    std::string engine_name, const platform::Place place)
    : batch_size_(batch_size), engine_name_(engine_name) {
  int i = 0;
  VLOG(4) << "Init a new calibrator: " << engine_name_;
  for (const auto it : buffers) {
    framework::Tensor temp_tensor;
    std::string input_name = it.first;
    int data_size = it.second;
    int num_ele = data_size / sizeof(int16_t);
    framework::DDim data_shape = framework::make_ddim({num_ele});
    temp_tensor.Resize(data_shape);
    data_tensors_.push_back(temp_tensor);
    data_buffers_[input_name] = std::pair<void*, size_t>(
        static_cast<void*>(temp_tensor.mutable_data<int16_t>(place)),
        data_size);
    i += 1;
  }
}

TRTInt8Calibrator::TRTInt8Calibrator(const std::string& calib_data)
    : batch_size_(0),
      calib_running_(false),
      data_is_set_(false),
      done_(true),
      calibration_table_(calib_data) {}

void TRTInt8Calibrator::waitAndSetDone() {
  std::unique_lock<std::mutex> lk(mut_);
  while ((calib_running_ || data_is_set_) && !done_) cond_.wait(lk);
  if (!done_) {
    done_ = true;
    cond_.notify_all();
  }
}

// There might be more than one input for trt subgraph,
// So, we use a map to store input information.
bool TRTInt8Calibrator::setBatch(
    const std::unordered_map<std::string, void*>& data) {
  VLOG(3) << "set batch: " << engine_name_;
  std::unique_lock<std::mutex> lk(mut_);
  //  There is a producer and a consumer. The producer set the batch data and
  //  the consumer get the batch data. The size of the data pool is one.
  //  So, the producer has to wait for the consumer to finish processing before
  //  they can set the data.
  while ((calib_running_ || data_is_set_) && (!done_)) cond_.wait(lk);
  // The done_ is set to true using waitAndSetDone, When all calibration data
  // are processed.
  if (done_) return false;

  // Sets the batch.
  for (const auto& it : data) {
    auto dataptr = data_buffers_.find(it.first);
    if (dataptr == data_buffers_.end()) {
      PADDLE_THROW(platform::errors::Fatal(
          "%s input name '%s' does not match with the buffer names.",
          engine_name_, it.first));
    }
    const auto& d = dataptr->second;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(d.first, it.second, d.second, cudaMemcpyDeviceToDevice));
  }

  data_is_set_ = true;
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) TRT_NOEXCEPT {
  VLOG(4) << "get batch: " << engine_name_;
  std::unique_lock<std::mutex> lk(mut_);
  // The consumer has just finished processing a data.
  // The producer can set the data again.
  calib_running_ = false;
  cond_.notify_all();

  // As long as there is data in the pool, the consumer can get it.
  while (!data_is_set_ && !done_) cond_.wait(lk);
  if (done_) return false;

  // Gets the batch
  for (int i = 0; i < num_bindings; i++) {
    auto it = data_buffers_.find(names[i]);
    if (it == data_buffers_.end()) {
      PADDLE_THROW(
          platform::errors::Fatal("Calibration engine asked for unknown tensor "
                                  "name '%s' at position %d.",
                                  names[i], i));
    }
    bindings[i] = it->second.first;
  }

  data_is_set_ = false;
  calib_running_ = true;
  VLOG(4) << "get batch done: " << engine_name_;
  return true;
}

void TRTInt8Calibrator::setDone() {
  std::unique_lock<std::mutex> lk(mut_);
  done_ = true;
  cond_.notify_all();
}

const void* TRTInt8Calibrator::readCalibrationCache(size_t& length)
    TRT_NOEXCEPT {
  if (calibration_table_.empty()) return nullptr;
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) TRT_NOEXCEPT {
  calibration_table_ = std::string((const char*)ptr, length);
  VLOG(4) << "Got calibration data for " << engine_name_ << " " << ptr
          << " length=" << length;
}
TRTInt8Calibrator::~TRTInt8Calibrator() {
  VLOG(4) << "Destroying calibrator for " << engine_name_;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
