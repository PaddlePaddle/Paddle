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


#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace inference {
namespace tensorrt {

struct TRTInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator {
 public:
  TRTInt8Calibrator(const std::unordered_map<std::string, size_t>& dev_buffers,
                    int batch_size, std::string engine_name,
                    const platform::Place place);

  TRTInt8Calibrator(const std::string& calibration_data);
  ~TRTInt8Calibrator();

  int getBatchSize() const override;

  bool getBatch(void* bindings[], const char* names[],
                int num_bindings) override;

  bool setBatch(const std::unordered_map<std::string, void*>& data);
  void setDone();
  void waitAndSetDone();

  const void* readCalibrationCache(std::size_t& length) override;
  void writeCalibrationCache(const void* ptr, std::size_t length) override;
  const std::string& getCalibrationTableAsString() {
    return calibration_table_;
  }

 private:
  const int batch_size_;

  bool calib_running_;
  bool data_is_set_;
  bool done_;

  std::mutex mut_;
  std::condition_variable cond_;

  std::unordered_map<std::string, std::pair<void*, size_t>> data_buffers_;
  std::vector<framework::Tensor> data_tensors_;

  std::string engine_name_;
  std::string calibration_table_;
};

class TRT_CalibratorRes {
 public:
  TRT_CalibratorRes() {}
  std::unique_ptr<TRTInt8Calibrator> calib_;
  std::unique_ptr<std::thread> thr_;
};
/*
 * Manager to control the TensorRT Int8 calibration creation and deltetion.
 */
class TRT_CalibratorResManager {
 public:
  bool Has(const std::string& name) const {
    if (res_.count(name) == 0) return false;
    return res_.at(name).get() != nullptr;
  }

  // Get Int8Calibrator via name
  TRT_CalibratorRes* Get(const std::string& name) const {
    return res_.at(name).get();
  }

  // Look up or create a calibrator.
  TRT_CalibratorRes* LookupOrCreate(std::string engine_name) {
    if (res_.count(engine_name) == 0) {
      auto* p = new TRT_CalibratorRes();
      res_[engine_name].reset(p);
    }
    return res_.at(engine_name).get();
  }

  // Create an Int8Calibrator
  TRT_CalibratorRes* Create(std::string engine_name) {
    auto* p = new TRT_CalibratorRes();
    res_[engine_name].reset(p);
    return p;
  }

  void DeleteALL() {
    for (auto& item : res_) {
      item.second.reset(nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<TRT_CalibratorRes>> res_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespacd paddle
